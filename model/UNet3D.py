#3D Unet with pos embedding

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
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

class Swish3D(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding3D(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.linear1 = nn.Linear(d_model, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = Swish3D()

    def forward(self, x):
        x = self.pe[:, x, :]
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

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

class KANActivation3D(nn.Module):
    """B样条基激活函数"""
    def __init__(self, grid_size=5, order=3, init_scale=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.order = order
        self.symbolic_mode = False
        
        # 初始化控制点，使用float32
        self.coefficients = nn.Parameter(torch.randn(grid_size + order, dtype=torch.float32) * init_scale)
        
        # 初始化节点向量，使用float32
        knots = torch.linspace(-2, 2, grid_size + 2 * order, dtype=torch.float32)
        self.register_buffer('knots', knots)
        
        # 用于符号化的表达式
        self.symbolic_expr = None
        
        # 预计算基函数的缓存
        self.basis_cache = {}
        
    def forward(self, x):
        if self.symbolic_mode and self.symbolic_expr is not None:
            return self.symbolic_expr(x)
        
        # 确保输入是float32
        x = x.float()
        result = torch.zeros_like(x, dtype=torch.float32)
        
        # 计算所有基函数
        basis_functions = self.compute_basis_functions(x)
        
        # 线性组合
        for i in range(len(self.coefficients)):
            result += self.coefficients[i] * basis_functions[i]
            
        return result
    
    def compute_basis_functions(self, x):
        """计算所有基函数"""
        n = len(self.coefficients)
        basis_functions = []
        
        # 计算0阶基函数
        N = [[torch.zeros_like(x, dtype=torch.float32) for _ in range(n + self.order)] 
             for _ in range(self.order + 1)]
        
        # 初始化0阶基函数
        for i in range(n + self.order - 1):
            N[0][i] = torch.where(
                (self.knots[i] <= x) & (x < self.knots[i + 1]),
                torch.ones_like(x, dtype=torch.float32),
                torch.zeros_like(x, dtype=torch.float32)
            )
        
        # 使用de Boor递推公式计算高阶基函数
        for k in range(1, self.order + 1):
            for i in range(n + self.order - k - 1):
                N[k][i] = torch.zeros_like(x, dtype=torch.float32)
                
                # 第一项
                if self.knots[i + k] != self.knots[i]:
                    w1 = (x - self.knots[i]) / (self.knots[i + k] - self.knots[i])
                    N[k][i] += w1 * N[k-1][i]
                
                # 第二项
                if self.knots[i + k + 1] != self.knots[i + 1]:
                    w2 = (self.knots[i + k + 1] - x) / (self.knots[i + k + 1] - self.knots[i + 1])
                    N[k][i] += w2 * N[k-1][i + 1]
        
        # 返回最高阶基函数
        return N[self.order][:n]
    
    def update_grid(self, x_samples):
        """根据输入样本更新网格点"""
        with torch.no_grad():
            x_min, x_max = x_samples.min(), x_samples.max()
            margin = (x_max - x_min) * 0.1
            new_knots = torch.linspace(x_min - margin, x_max + margin, 
                                     self.grid_size + 2 * self.order,
                                     dtype=torch.float32)
            self.knots.copy_(new_knots)
            # 清除缓存
            self.basis_cache.clear()
    
    def to_symbolic(self, sympy_x=None):
        """转换为符号表达式"""
        try:
            import sympy as sp
            if sympy_x is None:
                sympy_x = sp.Symbol('x')
            
            expr = 0
            for i in range(len(self.coefficients)):
                coeff = float(self.coefficients[i].detach().cpu())
                # 简化的符号表达式，使用多项式拟合
                expr += coeff * (sympy_x ** i)
            
            self.symbolic_expr = sp.lambdify(sympy_x, expr, 'torch')
            self.symbolic_mode = True
            return expr
        except ImportError:
            print("Sympy not found. Symbolic conversion not available.")
            return None

class TokKANLinear3D(nn.Module):
    """
    TokKANLinear3D: 3D version of KANLinear
    Implements the base transformation with non-linear activation
    """
    def __init__(self, in_channel, out_channel, norm_groups=32):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # Main transformation
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, 1, 1, 1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channel, dtype=torch.float32))
        
        # 替换SiLU为KANActivation
        self.act = KANActivation3D(grid_size=5, order=3)
        
        # Scale factor
        self.scale = 1.0
        
        # 符号化模式标志
        self.symbolic_mode = False
        
        # 参数共享组
        self.param_group = None
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
    
    def share_parameters(self, group):
        """加入参数共享组"""
        self.param_group = group
        if hasattr(group, 'shared_weight'):
            self.weight = group.shared_weight
        if hasattr(group, 'shared_bias'):
            self.bias = group.shared_bias
    
    def to_symbolic(self):
        """转换为符号模式"""
        self.symbolic_mode = True
        self.act.to_symbolic()
    
    def update_grid(self, x_samples):
        """更新激活函数的网格"""
        self.act.update_grid(x_samples)

    def forward(self, x):
        # 确保输入是float32
        x = x.float()
        
        # 应用GroupNorm
        x = self.norm(x)
        
        # 线性变换
        x = F.conv3d(x, self.weight * self.scale, self.bias)
        
        # 如果在参数共享组中，使用共享参数
        if self.param_group is not None:
            if hasattr(self.param_group, 'scale'):
                x = x * self.param_group.scale
        
        # 应用激活函数
        x = self.act(x)
        
        return x

class KANParameterGroup:
    """KAN参数共享组，用于在多个KAN层之间共享参数"""
    def __init__(self, channels, init_scale=1.0):
        self.shared_scale = nn.Parameter(torch.ones(1) * init_scale)
        self.members = []
    
    def add_member(self, layer):
        """添加层到共享组"""
        self.members.append(layer)
        layer.share_parameters(self)
    
    def update_all_grids(self, x_samples):
        """更新所有成员的网格"""
        for member in self.members:
            member.update_grid(x_samples)
    
    def to_symbolic_all(self):
        """将所有成员转换为符号模式"""
        for member in self.members:
            member.to_symbolic()

class KANLayer3D(nn.Module):
    """
    KANLayer3D: Implements the complete KAN layer with multiple transformations
    """
    def __init__(self, in_channel, out_channel, *, noise_level_emb_dim=None, norm_groups=32):
        super().__init__()
        
        # Noise level embedding
        self.noise_func = None
        if exists(noise_level_emb_dim):
            self.noise_func = nn.Sequential(
                nn.SiLU(),
                nn.Linear(noise_level_emb_dim, out_channel)
            )
        
        # First KAN block
        self.kan1 = TokKANLinear3D(in_channel, out_channel, norm_groups=norm_groups)
        
        # Second KAN block
        self.kan2 = TokKANLinear3D(out_channel, out_channel, norm_groups=norm_groups)
        
        # Third KAN block
        self.kan3 = TokKANLinear3D(out_channel, out_channel, norm_groups=norm_groups)
        
        # Residual connection
        self.res_conv = nn.Conv3d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()
        
        # 参数共享组
        self.param_group = None
        
        # 符号化模式标志
        self.symbolic_mode = False
    
    def share_parameters(self, group):
        """加入参数共享组"""
        self.param_group = group
        self.kan1.share_parameters(group)
        self.kan2.share_parameters(group)
        self.kan3.share_parameters(group)
    
    def to_symbolic(self):
        """转换为符号模式"""
        self.symbolic_mode = True
        self.kan1.to_symbolic()
        self.kan2.to_symbolic()
        self.kan3.to_symbolic()
    
    def update_grid(self, x_samples):
        """更新网格点"""
        self.kan1.update_grid(x_samples)
        self.kan2.update_grid(x_samples)
        self.kan3.update_grid(x_samples)

    def forward(self, x, time_emb=None):
        identity = self.res_conv(x)
        
        # First transformation
        h = self.kan1(x)
        
        # Add noise level embedding if available
        if exists(self.noise_func) and exists(time_emb):
            # time_emb shape: [B, 1, C] -> [B, C]
            time_emb = time_emb.squeeze(1)
            time_emb = self.noise_func(time_emb)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Second and third transformations
        h = self.kan2(h)
        h = self.kan3(h)
        
        # Residual connection
        return h + identity

class TokKANBlock3D(nn.Module):
    """
    TokKANBlock3D: Top level module that uses KANLayer3D
    """
    def __init__(self, in_channel, out_channel, *, noise_level_emb_dim=None, norm_groups=32, dropout=0.):
        super().__init__()
        self.kan = KANLayer3D(in_channel, out_channel, noise_level_emb_dim=noise_level_emb_dim, norm_groups=norm_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb=None):
        x = self.kan(x, time_emb)
        return self.dropout(x)

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

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None):
        """保存模型检查点
        Args:
            path: 保存路径
            epoch: 当前训练轮数
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """加载模型检查点
        Args:
            path: 检查点文件路径
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        Returns:
            last_epoch: 上次训练的轮数
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']

    def forward(self, x, t):
        if self.noise_level_mlp is not None:
            temb = self.noise_level_mlp(t)
        else:
            temb = None
        
        feats = []
        for layer in self.downs:
            if isinstance(layer, (ResnetBlocWithAttn, TokKANBlock3D)):
                x = layer(x, temb)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, temb)

        for layer in self.ups:
            if isinstance(layer, (Upsample, Downsample)):
                x = layer(x)
            else:
                x = torch.cat((x, feats.pop()), dim=1)
                x = layer(x, temb)

        return self.final_conv(x)



# 仿照segukan测试

class GELU3D(nn.Module):
    """
    GELU激活函数的3D版本
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class DropPath3D(nn.Module):
    """
    随机深度的3D版本
    """
    def __init__(self, drop_prob=None):
        super(DropPath3D, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp3D(nn.Module):
    """
    MLP模块的3D版本
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU3D, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        ):
        super(KANLinear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 初始化网格点
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        # 初始化权重 - 注意这里使用5D张量以支持3D数据
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1, 1, dtype=torch.float32))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order, 1, 1, 1)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features, 1, 1, 1)
            )
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        if self.enable_standalone_scale_spline:
            nn.init.ones_(self.spline_scaler)
    
    def b_splines(self, x):
        N = [[torch.zeros_like(x) for _ in range(self.grid_size + self.spline_order)] 
             for _ in range(self.spline_order + 1)]
        
        # 初始化0阶基函数
        for i in range(self.grid_size + self.spline_order - 1):
            N[0][i] = torch.where(
                (self.grid[:, i:i+1] <= x) & (x < self.grid[:, i+1:i+2]),
                torch.ones_like(x),
                torch.zeros_like(x)
            )
        
        # 使用de Boor递推公式计算高阶基函数
        for k in range(1, self.spline_order + 1):
            for i in range(self.grid_size + self.spline_order - k):
                w1 = torch.where(
                    self.grid[:, i+k:i+k+1] != self.grid[:, i:i+1],
                    (x - self.grid[:, i:i+1]) / (self.grid[:, i+k:i+k+1] - self.grid[:, i:i+1]),
                    torch.zeros_like(x)
                )
                w2 = torch.where(
                    self.grid[:, i+k+1:i+k+2] != self.grid[:, i+1:i+2],
                    (self.grid[:, i+k+1:i+k+2] - x) / (self.grid[:, i+k+1:i+k+2] - self.grid[:, i+1:i+2]),
                    torch.zeros_like(x)
                )
                N[k][i] = w1 * N[k-1][i] + w2 * N[k-1][i+1]
        
        return torch.stack(N[self.spline_order][:self.grid_size + self.spline_order], dim=2)
    
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(2)
        return self.spline_weight
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # 添加噪声
        if self.training and self.scale_noise > 0:
            noise = torch.randn_like(x) * self.scale_noise
            x = x + noise
        
        # 基础线性变换
        base_out = F.conv3d(x, self.base_weight * self.scale_base)
        base_out = self.base_activation(base_out)
        
        # B样条激活 - 处理3D数据
        x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W, C)
        spline_bases = self.b_splines(x_reshaped)  # (B*D*H*W, C, G)
        spline_out = torch.einsum('bci,oicjkl->bojkl', 
                                spline_bases.reshape(B, D*H*W, C, -1).permute(0, 2, 1, 3), 
                                self.scaled_spline_weight())
        spline_out = spline_out.reshape(B, self.out_features, D, H, W) * self.scale_spline
        
        return base_out + spline_out
    
    def update_grid(self, x, margin=0.01):
        with torch.no_grad():
            x_min = x.min().item()
            x_max = x.max().item()
            margin = (x_max - x_min) * margin
            h = (x_max - x_min + 2 * margin) / self.grid_size
            new_grid = (
                (torch.arange(-self.spline_order, self.grid_size + self.spline_order + 1) 
                 * h + (x_min - margin))
                .expand(self.in_features, -1)
                .to(self.grid.device)
            )
            self.grid.copy_(new_grid)

class KANBlock3D(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath3D(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, temb=None):
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, DHW, C
        x = x + self.drop_path(self.layer(self.norm2(x), D, H, W))
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x

class KANLayer3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear3D(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear3D(
                        hidden_features,
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
            self.fc3 = KANLinear3D(
                        hidden_features,
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
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu3D(hidden_features)
        self.dwconv_2 = DW_bn_relu3D(hidden_features)
        self.dwconv_3 = DW_bn_relu3D(hidden_features)
    
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x1 = self.dwconv_1(x)
        x2 = self.dwconv_2(x)
        x3 = self.dwconv_3(x)
        x = x.flatten(2).transpose(1, 2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = x3.flatten(2).transpose(1, 2)
        
        x1 = self.fc2(x1)
        x2 = self.fc3(x2)
        x = x1 + x2 + x3
        x = self.drop(x)
        return x

class DW_bn_relu3D(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.dwconv(x)))

class UKAN3D_NEW(nn.Module):
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

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None):
        """保存模型检查点
        Args:
            path: 保存路径
            epoch: 当前训练轮数
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """加载模型检查点
        Args:
            path: 检查点文件路径
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        Returns:
            last_epoch: 上次训练的轮数
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']

    def forward(self, x, t):
        if self.noise_level_mlp is not None:
            temb = self.noise_level_mlp(t)
        else:
            temb = None
        
        feats = []
        for layer in self.downs:
            if isinstance(layer, (ResnetBlocWithAttn, TokKANBlock3D)):
                x = layer(x, temb)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, temb)

        for layer in self.ups:
            if isinstance(layer, (Upsample, Downsample)):
                x = layer(x)
            else:
                x = torch.cat((x, feats.pop()), dim=1)
                x = layer(x, temb)

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

    def save_checkpoint(self, path, epoch, optimizer, scheduler=None):
        """保存模型检查点
        Args:
            path: 保存路径
            epoch: 当前训练轮数
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """加载模型检查点
        Args:
            path: 检查点文件路径
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        Returns:
            last_epoch: 上次训练的轮数
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']

    def forward(self, x, t):
        if self.noise_level_mlp is not None:
            temb = self.noise_level_mlp(t)
        else:
            temb = None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, temb)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, temb)

        for layer in self.ups:
            #connections from downsampling and upsamling.
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), temb)
            else:
                x = layer(x)

        return self.final_conv(x)
