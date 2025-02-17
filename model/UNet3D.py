#3D Unet with pos embedding

import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import torch.nn.functional as F

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        """
        :param noise_level: B*C
        :return:
        """
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) \
                   * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1,1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1,1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        #dim specifies the chanels
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        #nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)
        self.down = nn.MaxPool3d(kernel_size=2)
    def forward(self, x):
        return self.conv(self.down(x))

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        #dim specifies channels
        #dim_out specifies channels
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out,
                 noise_level_emb_dim=None, dropout=0,
                 use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        else:
            self.noise_func =None
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w,l = x.shape
        h = self.block1(x)
        if time_emb is not None and self.noise_func is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, length = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width,length)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwl, bncyxz -> bnhwlyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, length, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, length, height, width,length)

        out = torch.einsum("bnhwlyxz, bncyxz -> bnchwl", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width,length))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class TokKANLinear3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        norm_groups=32,
        bias=True
    ):
        super(TokKANLinear3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.padding = kernel_size // 2
        
        # Group Normalization
        self.norm = nn.GroupNorm(norm_groups, in_channels)
        
        # Base convolution path
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.base_activation = base_activation()
        
        # Spline path
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_channels, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        # Simplified spline path
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, 1, 1, 1)
        )
        
        self.scale_base = scale_base
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5) * self.scale_base)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize spline weights
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x):
        # Apply normalization
        x = self.norm(x)
        
        # Base convolution path
        x_base = self.base_activation(x)
        base_output = F.conv3d(x_base, self.weight, self.bias, padding=self.padding)
        
        # Simplified spline path
        spline_output = F.conv3d(x, self.spline_weight, None, padding=0)
        
        return base_output + spline_output

class TokKANBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.noise_func = None
        if exists(noise_level_emb_dim):
            self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out)

        # TokKAN path
        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.tokkan = TokKANLinear3D(dim, dim_out, norm_groups=norm_groups)
        self.norm2 = nn.GroupNorm(norm_groups, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = self.tokkan(h)
        h = self.norm2(h)
        h = self.dropout(h)

        if exists(self.noise_func) and exists(time_emb):
            h = self.noise_func(h, time_emb)

        return h + self.shortcut(x)

class UKAN3D(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        box_size=64,
    ):
        super().__init__()
        
        # 时间嵌入部分，与UNet3D保持一致
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = box_size
        
        # Initial conv
        downs = [nn.Conv3d(in_channel, inner_channel, kernel_size=3, padding=1)]
        
        # Downsampling path
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            
            # 在enc5层（最后一层）使用TokKAN
            if ind == num_mults - 1:
                for _ in range(0, res_blocks):
                    downs.append(TokKANBlock3D(
                        pre_channel, channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout
                    ))
                    feat_channels.append(channel_mult)
                    pre_channel = channel_mult
            else:
                for _ in range(0, res_blocks):
                    downs.append(ResnetBlocWithAttn(
                        pre_channel, channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn
                    ))
                    feat_channels.append(channel_mult)
                    pre_channel = channel_mult
            
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        
        self.downs = nn.ModuleList(downs)
        
        # Middle blocks
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel,
                             noise_level_emb_dim=noise_level_channel,
                             norm_groups=norm_groups,
                             dropout=dropout,
                             with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                             noise_level_emb_dim=noise_level_channel,
                             norm_groups=norm_groups,
                             dropout=dropout,
                             with_attn=False)
        ])
        
        # Upsampling path
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            
            # 在dec1层（对应最深层的上采样）使用TokKAN
            if ind == num_mults - 1:
                for _ in range(0, res_blocks+1):
                    ups.append(TokKANBlock3D(
                        pre_channel+feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout
                    ))
                    pre_channel = channel_mult
            else:
                for _ in range(0, res_blocks+1):
                    ups.append(ResnetBlocWithAttn(
                        pre_channel+feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn
                    ))
                    pre_channel = channel_mult
            
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2
        
        self.ups = nn.ModuleList(ups)
        
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time=None):
        """
        x: (B, C, D, H, W)
        time: (B, 1)
        """
        if exists(self.noise_level_mlp) and exists(time):
            time_emb = self.noise_level_mlp(time)
        else:
            time_emb = None

        feats = []
        for layer in self.downs:
            if isinstance(layer, (ResnetBlocWithAttn, TokKANBlock3D)):
                x = layer(x, time_emb)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.ups:
            if isinstance(layer, (Upsample, Downsample)):
                x = layer(x)
            else:
                x = torch.cat((x, feats.pop()), dim=1)
                x = layer(x, time_emb)

        return self.final_conv(x)

class UNet3D(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        box_size=64,
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = box_size
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        #configure downsampling layers
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            #continue double channels
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time=None):
        #time here is the gamma
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            #connections from downsampling and upsamling.
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

# class UKAN3D(nn.Module):
#     def __init__(
#         self,
#         in_channel=2,
#         out_channel=1,
#         inner_channel=32,
#         norm_groups=32,
#         channel_mults=(1, 2, 4, 8, 8),
#         attn_res=(8),
#         res_blocks=3,
#         dropout=0,
#         with_noise_level_emb=True,
#         box_size=64,
#     ):
#         super().__init__()
#         if with_noise_level_emb:
#             noise_level_channel = inner_channel
#             self.noise_level_mlp = nn.Sequential(
#                 PositionalEncoding(inner_channel),
#                 nn.Linear(inner_channel, inner_channel * 4),
#                 Swish(),
#                 nn.Linear(inner_channel * 4, inner_channel)
#             )
#         else:
#             noise_level_channel = None
#             self.noise_level_mlp = None
            
#         num_mults = len(channel_mults)
#         pre_channel = inner_channel
#         feat_channels = [pre_channel]
#         now_res = box_size
        
#         # Initial conv
#         downs = [nn.Conv3d(in_channel, inner_channel, kernel_size=3, padding=1)]
        
#         # Configure downsampling layers
#         for ind in range(num_mults):
#             is_last = (ind == num_mults - 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = inner_channel * channel_mults[ind]
            
#             # For the last two scales, use KANBlock3D
#             if ind >= num_mults - 2:  # 最后两个尺度使用KANBlock3D
#                 for _ in range(0, res_blocks):
#                     downs.append(KANBlock3D(
#                         pre_channel, channel_mult,
#                         noise_level_emb_dim=noise_level_channel,
#                         norm_groups=norm_groups,
#                         dropout=dropout
#                     ))
#                     feat_channels.append(channel_mult)
#                     pre_channel = channel_mult
#             else:  # 其他尺度使用原来的ResnetBlocWithAttn
#                 for _ in range(0, res_blocks):
#                     downs.append(ResnetBlocWithAttn(
#                         pre_channel, channel_mult,
#                         noise_level_emb_dim=noise_level_channel,
#                         norm_groups=norm_groups,
#                         dropout=dropout,
#                         with_attn=use_attn
#                     ))
#                     feat_channels.append(channel_mult)
#                     pre_channel = channel_mult
                    
#             if not is_last:
#                 downs.append(Downsample(pre_channel))
#                 feat_channels.append(pre_channel)
#                 now_res = now_res//2
        
#         self.downs = nn.ModuleList(downs)
        
#         # Middle blocks with KANBlock3D
#         self.mid = nn.ModuleList([
#             KANBlock3D(pre_channel, pre_channel,
#                       noise_level_emb_dim=noise_level_channel,
#                       norm_groups=norm_groups,
#                       dropout=dropout),
#             KANBlock3D(pre_channel, pre_channel,
#                       noise_level_emb_dim=noise_level_channel,
#                       norm_groups=norm_groups,
#                       dropout=dropout)
#         ])
        
#         # Upsampling path
#         ups = []
#         for ind in reversed(range(num_mults)):
#             is_last = (ind < 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = inner_channel * channel_mults[ind]
            
#             # For the first two scales (corresponding to the last two in downsampling), use KANBlock3D
#             if ind >= num_mults - 2:
#                 for _ in range(0, res_blocks+1):
#                     ups.append(KANBlock3D(
#                         pre_channel+feat_channels.pop(),
#                         channel_mult,
#                         noise_level_emb_dim=noise_level_channel,
#                         norm_groups=norm_groups,
#                         dropout=dropout
#                     ))
#                     pre_channel = channel_mult
#             else:
#                 for _ in range(0, res_blocks+1):
#                     ups.append(ResnetBlocWithAttn(
#                         pre_channel+feat_channels.pop(),
#                         channel_mult,
#                         noise_level_emb_dim=noise_level_channel,
#                         norm_groups=norm_groups,
#                         dropout=dropout,
#                         with_attn=use_attn
#                     ))
#                     pre_channel = channel_mult
                    
#             if not is_last:
#                 ups.append(Upsample(pre_channel))
#                 now_res = now_res*2
        
#         self.ups = nn.ModuleList(ups)
        
#         self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

#     def forward(self, x, time=None):
#         """
#         x: (B, C, D, H, W)
#         time: (B, 1)
#         """
#         t = self.noise_level_mlp(time) if exists(time) else None

#         feats = []
#         for layer in self.downs:
#             if isinstance(layer, (ResnetBlocWithAttn, KANBlock3D)):
#                 x = layer(x, t)
#             else:
#                 x = layer(x)
#             feats.append(x)

#         for layer in self.mid:
#             x = layer(x, t)

#         for layer in self.ups:
#             if isinstance(layer, Upsample):
#                 x = layer(x)
#             else:
#                 x = layer(torch.cat((x, feats.pop()), dim=1), t)

#         return self.final_conv(x)

# 新的尝试


# class TokKANLinear3D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         enable_standalone_scale_spline=True,
#         base_activation=nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(TokKANLinear3D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.grid_size = grid_size
#         self.spline_order = spline_order
#         self.padding = kernel_size // 2

#         h = (grid_range[1] - grid_range[0]) / grid_size
#         grid = (
#             (
#                 torch.arange(-spline_order, grid_size + spline_order + 1) * h
#                 + grid_range[0]
#             )
#             .expand(in_channels, -1)
#             .contiguous()
#         )
#         self.register_buffer("grid", grid)

#         self.base_conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=self.padding)
#         self.spline_weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, grid_size + spline_order)
#         )
#         if enable_standalone_scale_spline:
#             self.spline_scaler = nn.Parameter(
#                 torch.Tensor(out_channels, in_channels)
#             )

#         self.scale_noise = scale_noise
#         self.scale_base = scale_base
#         self.scale_spline = scale_spline
#         self.enable_standalone_scale_spline = enable_standalone_scale_spline
#         self.base_activation = base_activation()
#         self.grid_eps = grid_eps

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.base_conv.weight, a=math.sqrt(5) * self.scale_base)
#         with torch.no_grad():
#             noise = (
#                 (
#                     torch.rand(self.grid_size + 1, self.in_channels, self.out_channels)
#                     - 1 / 2
#                 )
#                 * self.scale_noise
#                 / self.grid_size
#             )
#             self.spline_weight.data.copy_(
#                 (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
#                 * self.curve2coeff(
#                     self.grid.T[self.spline_order : -self.spline_order],
#                     noise,
#                 )
#             )
#             if self.enable_standalone_scale_spline:
#                 nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

#     def curve2coeff(self, x, y):
#         """Convert curve points to B-spline coefficients."""
#         n = x.size(0)
#         A = torch.zeros(n, n, device=x.device)
#         for i in range(n):
#             t = x[i]
#             bases = self.b_splines(t.expand(1, -1))
#             A[i] = bases[0, 0]
#         return torch.linalg.solve(A, y.permute(1, 2, 0)).permute(2, 0, 1)

#     def b_splines(self, x):
#         """Compute B-spline bases."""
#         grid = self.grid
#         x = x.unsqueeze(-1)
#         bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
#         for k in range(1, self.spline_order + 1):
#             bases = (
#                 (x - grid[:, : -(k + 1)])
#                 / (grid[:, k:-1] - grid[:, : -(k + 1)])
#                 * bases[:, :, :-1]
#             ) + (
#                 (grid[:, k + 1 :] - x)
#                 / (grid[:, k + 1 :] - grid[:, 1 : -(k)])
#                 * bases[:, :, 1:]
#             )
#         return bases

#     def forward(self, x):
#         # Base convolution path
#         base_out = self.base_conv(x)
#         base_out = self.base_activation(base_out)

#         # Spline path
#         B, C, D, H, W = x.shape
#         x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, D*H*W, C]
        
#         bases = self.b_splines(x_flat)  # [B, D*H*W, C, grid_size + spline_order]
        
#         if self.enable_standalone_scale_spline:
#             spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
#         else:
#             spline_weight = self.spline_weight
            
#         spline_out = torch.einsum('bdhwc,cod->bdhwo', bases, spline_weight)
#         spline_out = spline_out.view(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)

#         return base_out + spline_out


# class TokKAN3D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#         norm_groups=32,
#     ):
#         super(TokKAN3D, self).__init__()
#         self.norm = nn.GroupNorm(norm_groups, in_channels)
#         self.kan = TokKANLinear3D(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             grid_size=grid_size,
#             spline_order=spline_order,
#             scale_noise=scale_noise,
#             scale_base=scale_base,
#             scale_spline=scale_spline,
#             base_activation=base_activation,
#             grid_eps=grid_eps,
#             grid_range=grid_range,
#         )

#     def forward(self, x):
#         x = self.norm(x)
#         return self.kan(x)


# class DWConv3D(nn.Module):
#     def __init__(self, dim):
#         super(DWConv3D, self).__init__()
#         self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#         self.norm = nn.GroupNorm(32, dim)
#         self.act = Swish()

#     def forward(self, x):
#         return self.act(self.norm(self.dwconv(x)))

# class KANLayer3D(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0):
#         super().__init__()
#         self.hidden_features = hidden_features or in_features
#         self.out_features = out_features or in_features
        
#         # First KAN path
#         self.kan1 = TokKANLinear3D(in_features, self.hidden_features)
#         self.dwconv1 = DWConv3D(self.hidden_features)
        
#         # Second KAN path
#         self.kan2 = TokKANLinear3D(self.hidden_features, self.out_features)
#         self.dwconv2 = DWConv3D(self.hidden_features)
        
#         # Third KAN path
#         self.kan3 = TokKANLinear3D(self.hidden_features, self.out_features)
#         self.dwconv3 = DWConv3D(self.hidden_features)
        
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         # First KAN block
#         x = self.kan1(x)
#         x = self.dwconv1(x)
#         x = self.drop(x)
        
#         # Second KAN block
#         x = self.kan2(x)
#         x = self.dwconv2(x)
#         x = self.drop(x)
        
#         # Third KAN block
#         x = self.kan3(x)
#         x = self.dwconv3(x)
#         x = self.drop(x)
        
#         return x

# class KANBlock3D(nn.Module):
#     def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, dropout=0, norm_groups=32):
#         super().__init__()
#         self.noise_func = None
#         if exists(noise_level_emb_dim):
#             self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out)

#         self.norm1 = nn.GroupNorm(norm_groups, dim)
#         self.kan = KANLayer3D(dim, dim_out, dim_out, dropout=dropout)
#         self.norm2 = nn.GroupNorm(norm_groups, dim_out)
#         self.shortcut = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         h = self.norm1(x)
#         h = self.kan(h)
#         h = self.norm2(h)

#         if exists(self.noise_func):
#             h = self.noise_func(h, time_emb)

#         return h + self.shortcut(x)
