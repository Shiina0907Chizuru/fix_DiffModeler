# 网络接口文件对照表

## 1. 文件结构对比

### 旧版DDIM实现
```
model/
├── DDIM.py                # DDIM模型实现
├── Base_DDIM.py          # DDIM基础类
├── config_networks.py     # 网络配置和创建
└── networks/
    └── UNet3D.py         # UNet网络实现
```

### 新版通用接口
```
model/
├── NetworkTemplate.py     # 通用网络模板
├── BaseNetwork.py        # 基础网络类
├── config_networks_new.py # 新版网络配置和创建
└── networks/
    ├── UNet3D.py         # UNet网络实现
    └── CustomNet.py      # 自定义网络示例
```

## 2. 核心文件对应关系

| 旧版文件 | 新版文件 | 主要变化 |
|---------|---------|---------|
| `Base_DDIM.py` | `BaseNetwork.py` | 移除DDIM特定实现，提供通用网络基类 |
| `DDIM.py` | `NetworkTemplate.py` | 简化为通用网络模板 |
| `config_networks.py` | `config_networks_new.py` | 重构网络创建逻辑，支持多种网络类型 |

## 3. 主要接口对比

### 3.1 网络基类
```python
# 旧版 (Base_DDIM.py)
class Base_DDIM:
    def __init__(self, opt):
        self.setup_diffusion()  # DDIM特定
        
    def calculate_loss(self):
        # DDIM损失计算
        pass

# 新版 (BaseNetwork.py)
class BaseNetwork:
    def __init__(self, opt):
        self.setup_network()    # 通用网络设置
        
    def calculate_loss(self):
        # 通用损失计算接口
        pass
```

### 3.2 网络实现
```python
# 旧版 (DDIM.py)
class DDIM(Base_DDIM):
    def __init__(self, opt):
        self.netG = define_G(opt)
        self.setup_optimizers()
        
    def optimize_parameters(self):
        # DDIM特定优化
        pass

# 新版 (NetworkTemplate.py)
class NetworkTemplate(BaseNetwork):
    def __init__(self, opt):
        self.network = self._build_network()
        self.setup_optimizers()
        
    def optimize_parameters(self):
        # 通用优化接口
        pass
```

### 3.3 网络配置
```python
# 旧版 (config_networks.py)
def define_G(opt):
    # 仅支持UNet配置
    return UNet3D(opt['model']['unet'])

# 新版 (config_networks_new.py)
def define_network(opt):
    # 支持多种网络类型
    if opt['model']['type'] == 'unet':
        return UNet3D(opt['model']['network'])
    elif opt['model']['type'] == 'custom':
        return CustomNet(opt['model']['network'])
```

## 4. 配置文件对比

### 4.1 旧版配置 (diffmodeler.json)
```json
{
    "model": {
        "unet": {
            "norm_groups": 32,
            "inner_channel": 64,
            "channel_multiplier": [1, 2, 4, 8],
            "attn_res": [16],
            "res_blocks": 2,
            "dropout": 0.2
        }
    }
}
```

### 4.2 新版配置 (my_network.json)
```json
{
    "name": "custom_network",
    "model": {
        "type": "custom",
        "network": {
            "in_channels": 1,
            "hidden_channels": 64,
            "out_channels": 1,
            "num_blocks": 4
        }
    }
}
```

## 5. 主要改进点

1. **解耦合**
   - 旧版：紧密耦合DDIM实现
   - 新版：完全解耦的通用接口

2. **扩展性**
   - 旧版：难以添加新网络类型
   - 新版：易于添加和配置新网络

3. **配置系统**
   - 旧版：固定的UNet配置
   - 新版：灵活的网络配置结构

4. **代码组织**
   - 旧版：特定于DDIM的实现
   - 新版：清晰的模块化结构

## 6. 使用建议

1. **新网络开发**
   - 继承 `NetworkTemplate`
   - 实现必要的接口方法
   - 在 `config_networks_new.py` 中注册

2. **配置文件**
   - 使用新的配置格式
   - 明确指定网络类型
   - 合理组织网络参数

3. **迁移建议**
   - 保持旧版DDIM实现
   - 新网络使用新接口
   - 逐步迁移现有代码
