# 3D UMLP with KAN activation for DiffModeler

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from inspect import isfunction
from typing import List, Tuple, Union

from model.UNet3D import exists, default, PositionalEncoding, Swish, Upsample, Downsample, KANActivation3D, Block, ResnetBlocWithAttn

# Activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 3D version of OverlapPatchEmbed for KAN
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # B C D*H*W -> B D*H*W C
        x = self.norm(x)
        return x, D, H, W

class OverlapPatchEmbed3D(nn.Module):
    def __init__(self, vol_size=64, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=patch_size, 
                             stride=stride, 
                             padding=(patch_size//2, patch_size//2, patch_size//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, D, H, W = x.shape
        x = self.proj(x)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # B C D*H*W -> B D*H*W C
        x = self.norm(x)
        return x, D, H, W

# 3D version of KAN
class KANLinear3D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline) * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        # 确保输入维度正确
        original_shape = x.shape
        if x.dim() > 2:
            # 如果输入维度 > 2，把它展平为(batch_size, in_features)形式
            x = x.reshape(-1, self.in_features)
        
        assert x.size(-1) == self.in_features, f"输入特征维度为{x.size(-1)}，期望维度为{self.in_features}"

        base_output = self.base_activation(F.linear(x, self.base_weight))
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        result = base_output + spline_output
        
        # 如果需要，恢复原始维度
        if len(original_shape) > 2:
            result = result.view(*original_shape[:-1], self.out_features)
            
        return result

# KAN3D: 3D version of KAN
class KAN3D(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=Swish,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN3D, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear3D(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor, update_grid=False):
        for i, layer in enumerate(self.layers):
            if update_grid and i < len(self.layers) - 1:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        loss = 0.0
        if regularize_activation > 0:
            for layer in self.layers:
                # L2 regularization on the scaled spline weights
                loss = loss + regularize_activation * torch.mean(
                    torch.square(layer.scaled_spline_weight)
                )
        return loss

# 1x1x1 convolution
def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 3D patch embedding with overlapping
class OverlapPatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding with overlapping
    """
    def __init__(self, vol_size=64, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        vol_size = vol_size if isinstance(vol_size, tuple) else (vol_size, vol_size, vol_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.D = vol_size[0] // stride
        self.H = vol_size[1] // stride
        self.W = vol_size[2] // stride
        self.num_patches = self.D * self.H * self.W
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, D, H, W

# 3D down-sampling block
class DownSample3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x, temb):
        x = self.main(x)
        return x

# 3D up-sampling block
class UpSample3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x

# 3D KAN implementation for shifted window attention
class kan3d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kan_val=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=Swish
        grid_eps=0.02
        grid_range=[-1, 1]

        if kan_val:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                Swish(),
                nn.Linear(hidden_features, out_features))
            
        self.drop = nn.Dropout(drop)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
    
    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.drop(x)
        return x

# 3D Shifted block with KAN
class shiftedBlock3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, kan_val=False):
        super().__init__()

        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, dim),
        )
        self.kan = kan3d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop, kan_val=kan_val)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
    
    def forward(self, x, D, H, W, temb):
        # Add time embedding
        temb_proj = self.temb_proj(temb)[:, None, :]  # [B, 1, dim]
        x = x + temb_proj
        
        x = x + self.drop_path(self.kan(self.norm2(x), D, H, W))
        return x

# 3D Depthwise Convolution
class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    
    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# 3D Depthwise Conv with GroupNorm and ReLU
class DW3D_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW3D_bn_relu, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.GroupNorm(32, dim)
    
    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# 3D Single Convolution with time embedding
class SingleConv3D(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(SingleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, h_ch, 3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    
    def forward(self, input, temb):
        h = self.conv(input)
        temb = self.temb_proj(temb)[:, :, None, None, None]
        return h + temb

# 3D Double Convolution with time embedding
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Conv3d(h_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    
    def forward(self, input, temb):
        h = self.conv(input)
        temb = self.temb_proj(temb)[:, :, None, None, None]
        return h + temb

# Double Conv3D for decoder
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.act1 = Swish()
        
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = Swish()
        
        # Optional time embedding projection
        if time_emb_dim is not None:
            self.temb_proj = nn.Sequential(
                Swish(),
                nn.Linear(time_emb_dim, mid_channels)
            )
        else:
            self.temb_proj = None
            
    def forward(self, x, temb=None):
        x = self.conv1(x)
        x = self.norm1(x)
        
        if temb is not None and self.temb_proj is not None:
            x = x + self.temb_proj(temb)[:, :, None, None, None]
            
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x

# 3D Decoder Single Convolution
class D_SingleConv3D(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_SingleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, h_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    
    def forward(self, input, temb):
        h = self.conv(input)
        temb = self.temb_proj(temb)[:, :, None, None, None]
        return h + temb

# 3D Decoder Double Convolution
class D_DoubleConv3D(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    
    def forward(self, input, temb):
        h = self.conv(input)
        temb = self.temb_proj(temb)[:, :, None, None, None]
        return h + temb

# 3D Attention Block
class AttnBlock3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        q = q.view(B, C, D*H*W).permute(0, 2, 1)  # B, D*H*W, C
        k = k.view(B, C, D*H*W)  # B, C, D*H*W
        w = torch.bmm(q, k)  # B, D*H*W, D*H*W
        w = w * (int(C) ** (-0.5))
        w = F.softmax(w, dim=2)
        
        v = v.view(B, C, D*H*W)  # B, C, D*H*W
        h = torch.bmm(v, w.permute(0, 2, 1))  # B, C, D*H*W
        h = h.view(B, C, D, H, W)
        
        h = self.proj(h)
        
        return x + h

# 3D Residual Block
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, h_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, h_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(h_ch, h_ch, 3, stride=1, padding=1),
        )
        if in_ch != h_ch:
            self.shortcut = nn.Conv3d(in_ch, h_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock3D(h_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        try:
            init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        except:
            pass
    
    def forward(self, x, temb):
        h = self.block1(x)
        
        # 确保temb是二维张量 [B, C]
        if len(temb.shape) > 2:
            temb = temb.view(temb.shape[0], -1)
        
        # 检查temb的形状并适当调整
        temb_projected = self.temb_proj(temb)
        # 添加所需的维度以匹配h的形状
        temb_projected = temb_projected[:, :, None, None, None]
        
        # 确保维度匹配后再相加
        h = h + temb_projected
        h = self.block2(h)
        
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

# Block for KAN with MLP
class Block3DMLP(nn.Module):
    def __init__(self, dim, n_div=4, mlp_ratio=4, drop=0., mlp_fn=KANLinear3D, temb_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 时间嵌入投影，如果提供了temb_dim
        if temb_dim is not None:
            self.temb_proj = nn.Sequential(
                nn.Linear(temb_dim, dim),
                nn.GELU()
            )
        else:
            self.temb_proj = None
        
        # MLP层
        self.mlp = nn.Sequential(
            mlp_fn(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            mlp_fn(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, D, H, W, temb=None):
        # Layer Norm + MLP + Residual
        x = x + self.mlp(self.norm1(x))
        
        # 如果提供了时间嵌入并且有temb_proj
        if temb is not None and self.temb_proj is not None:
            # 确保时间嵌入的维度匹配
            temb_proj = self.temb_proj(temb)
            # 将时间嵌入调整为[B, 1, dim]格式，方便与x相加
            temb_proj = temb_proj[:, None, :]
            x = x + temb_proj
            
        return x

# 3D UMLP Module
class UMLP3D(nn.Module):
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
        ch = inner_channel
        ch_mult = channel_mults
        
        # 处理attn_res参数，确保它是一个序列
        if isinstance(attn_res, int):
            attn_res = (attn_res,)
        
        attn = [i for i, res in enumerate(channel_mults) if res in attn_res]  # 根据attn_res确定attn层
        num_res_blocks = res_blocks
        T = 1000  # 默认时间步
        
        tdim = ch * 4
        self.time_embedding = PositionalEncoding(tdim)
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            Swish(),
            nn.Linear(tdim, tdim)
        )
        
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.box_size = box_size
        
        # Initial convolution
        self.head = nn.Conv3d(in_channel, ch, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record input channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock3D(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample3D(now_ch))
                chs.append(now_ch)
        
        # Upsampling blocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock3D(
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
            if i != 0:
                self.upblocks.append(UpSample3D(now_ch))
        assert len(chs) == 0
        
        # Output convolution
        self.tail = nn.Sequential(
            nn.GroupNorm(norm_groups, now_ch),
            Swish(),
            nn.Conv3d(now_ch, out_channel, 3, stride=1, padding=1)
        )
        
        # KAN blocks configuration
        embed_dim = 256  # KAN嵌入维度
        depth = 3  # KAN块深度
        
        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed3D(vol_size=box_size // 4, patch_size=1, stride=1, in_chans=ch * ch_mult[-1], embed_dim=embed_dim)
        
        self.kan_block1 = nn.ModuleList([
            Block3DMLP(
                dim=embed_dim, n_div=1, mlp_ratio=2, drop=dropout,
                mlp_fn=KANLinear3D, temb_dim=tdim) for _ in range(depth)
        ])
        
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed3D(vol_size=box_size // 8, patch_size=2, stride=2, in_chans=embed_dim, embed_dim=embed_dim*2)
        
        self.kan_block2 = nn.ModuleList([
            Block3DMLP(
                dim=embed_dim*2, n_div=1, mlp_ratio=2, drop=dropout,
                mlp_fn=KANLinear3D, temb_dim=tdim) for _ in range(depth)
        ])
        
        self.norm4 = nn.LayerNorm(embed_dim*2)
        
        # Decoder Stage 4
        self.decoder1 = DoubleConv3D(embed_dim*2, embed_dim)
        
        # Decoder Stage 3
        self.kan_dblock1 = nn.ModuleList([
            Block3DMLP(
                dim=embed_dim, n_div=1, mlp_ratio=2, drop=dropout,
                mlp_fn=KANLinear3D, temb_dim=tdim) for _ in range(depth)
        ])
        
        self.dnorm3 = nn.LayerNorm(embed_dim)
        self.decoder2 = DoubleConv3D(embed_dim, ch * ch_mult[-1])
        
        # Initialize
        self.apply(self._init_weights)
        init.zeros_(self.tail[-1].weight)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
    
    def forward(self, x, time=None):
        """
        x: (B, C, D, H, W)
        time: (B, 1)
        """
        # 时间嵌入
        if time is None:
            # 如果没有提供时间信息，使用零张量
            batch_size = x.shape[0]
            time = torch.zeros((batch_size,), device=x.device)
            
        temb = self.time_embedding(time)
        temb = self.time_mlp(temb)
        
        # 下采样路径
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        
        # 保存中间结果用于跳跃连接
        skip_connection = h
        
        # KAN特性提取
        B, _, D, H, W = h.shape  # 获取当前特征图的尺寸
        h, D_out, H_out, W_out = self.patch_embed3(h)  # D_out, H_out, W_out是经过patch_embed后的空间尺寸
        
        # KAN处理
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, D_out, H_out, W_out, temb)
        h = self.norm3(h)
        
        # 重新整形为通道优先的格式 (B, C, D, H, W)
        h = h.reshape(B, D_out, H_out, W_out, -1).permute(0, 4, 1, 2, 3).contiguous()
        h_skip1 = h  # 保存用于跳跃连接
        
        # 继续KAN处理
        h, D_out2, H_out2, W_out2 = self.patch_embed4(h)
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, D_out2, H_out2, W_out2, temb)
        h = self.norm4(h)
        
        # 重新整形
        h = h.reshape(B, D_out2, H_out2, W_out2, -1).permute(0, 4, 1, 2, 3).contiguous()
        
        # 解码器阶段
        h = self.decoder1(h, temb)
        h = F.interpolate(h, size=(D_out, H_out, W_out), mode='trilinear', align_corners=False)
        
        # 跳跃连接
        h = torch.add(h, h_skip1)
        
        # 重整用于KAN块
        _, _, D_cur, H_cur, W_cur = h.shape
        h = h.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, D_cur, H_cur, W_cur, temb)
        
        # 再次解码
        h = self.dnorm3(h)
        h = h.reshape(B, D_cur, H_cur, W_cur, -1).permute(0, 4, 1, 2, 3).contiguous()
        h = self.decoder2(h, temb)
        
        # 上采样到与skip_connection相同的尺寸
        h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
        
        # 跳跃连接
        h = torch.add(h, skip_connection)
        
        # 上采样路径
        for layer in self.upblocks:
            if isinstance(layer, ResBlock3D):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        
        h = self.tail(h)
        
        assert len(hs) == 0
        return h

# Function to get the network class
def get_network_class(network_type):
    """Return the appropriate class network type."""
    if network_type == "umlp3d":
        return UMLP3D
    else:
        # For other network types, fall back to UNet3D's get_network_class
        from model.UNet3D import get_network_class as unet3d_get_network_class
        return unet3d_get_network_class(network_type)