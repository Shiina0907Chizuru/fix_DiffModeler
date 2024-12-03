# DiffModeler 网络实现指南

## 1. DDIM实现流程

### 1.1 入口点 (train.py)
```python
# train.py
from ops.argparser import argparser_train
from training.main_worker import main_worker

def main():
    # 1. 解析参数并加载配置
    params = argparser_train()
    # 2. 开始训练
    main_worker(params)
```

### 1.2 主要工作流程 (training/main_worker.py)
```python
def main_worker(params):
    # 1. 初始化路径
    log_path, model_path = init_save_path(params)
    
    # 2. 创建DDIM实例
    from model.DDIM import DDIM
    ddim_runner = DDIM(params)
    model = ddim_runner.netG
    optimizer = ddim_runner.optG
    
    # 3. 配置数据集
    train_loader, val_loader = configure_dataset(params)
    
    # 4. 训练循环
    for epoch in range(start_epoch, params['train']['epoch']):
        train_loss = train_diffmodeler(train_loader, ddim_runner, model, ...)
        val_loss = valid_diffmodeler(val_loader, ddim_runner, model, ...)
```

### 1.3 DDIM类实现 (model/DDIM.py)
```python
class DDIM(Base_DDIM):
    def __init__(self, opt):
        super(DDIM, self).__init__(opt)
        # 1. 创建网络
        self.netG = define_G(opt)
        
        # 2. 如果是训练阶段，设置优化器
        if self.opt['phase'] == 'train':
            self.optG = self.setup_optimizers()
            
    def feed_data(self, data):
        # 数据预处理
        self.real_H = data['target'].to(self.device)
        self.condition = data['condition'].to(self.device)
        
    def optimize_parameters(self):
        # 优化步骤
        self.optG.zero_grad()
        loss = self.calculate_loss()
        loss.backward()
        self.optG.step()
```

### 1.4 基础DDIM实现 (model/Base_DDIM.py)
```python
class Base_DDIM:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.setup_diffusion()
        
    def setup_diffusion(self):
        # 设置beta schedule
        self.beta_schedule = make_beta_schedule(...)
        
    def calculate_loss(self):
        # 实现扩散模型的损失计算
        t = torch.randint(...)
        noise = torch.randn_like(...)
        return loss
```

## 2. 新网络接口使用指南

### 2.1 基础网络模板
```python
class NetworkTemplate(BaseNetwork):
    def __init__(self, opt):
        super(NetworkTemplate, self).__init__(opt)
        self.network = self._build_network()
        if opt['phase'] == 'train':
            self.setup_optimizers()
            self._setup_loss()
    
    def feed_data(self, data):
        """输入数据预处理"""
        self.input = data['input'].to(self.device)
        self.target = data['target'].to(self.device)
    
    def optimize_parameters(self):
        """优化网络参数"""
        self.optimizer.zero_grad()
        self.output = self.network(self.input)
        loss = self.loss_func(self.output, self.target)
        loss.backward()
        self.optimizer.step()
```

### 2.2 创建新网络

1. 创建网络配置文件 (config/my_network.json):
```json
{
    "name": "my_network",
    "phase": "train",
    "model": {
        "type": "my_network",
        "network": {
            "in_channels": 1,
            "hidden_channels": 64,
            "out_channels": 1
        }
    },
    "train": {
        "optimizer": {
            "type": "Adam",
            "lr": 0.001
        }
    }
}
```

2. 实现网络类:
```python
class MyNetwork(NetworkTemplate):
    def _build_network(self):
        """构建自定义网络架构"""
        return nn.Sequential(
            nn.Conv3d(self.opt['network']['in_channels'],
                     self.opt['network']['hidden_channels'],
                     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.opt['network']['hidden_channels'],
                     self.opt['network']['out_channels'],
                     kernel_size=3, padding=1)
        )
    
    def _setup_loss(self):
        """设置自定义损失函数"""
        self.loss_func = nn.MSELoss()
```

3. 在config_networks_new.py中注册网络:
```python
def define_network(opt):
    net_type = opt['model']['type']
    if net_type == 'my_network':
        return MyNetwork(opt)
```

## 3. 新旧接口对比

### 3.1 主要改进
1. **简化接口**
   - 移除了DDIM特定的方法
   - 更通用的网络模板
   - 更清晰的关注点分离

2. **配置系统**
   - 更灵活的网络配置
   - 更容易添加新的网络类型
   - 更好的参数组织

3. **实现流程**
   - DDIM：特定的扩散模型实现
   - 新接口：通用网络框架
   - 更容易扩展和修改

### 3.2 使用建议
1. **网络实现**
   - 继承NetworkTemplate
   - 实现必要的方法
   - 保持配置清晰

2. **配置管理**
   - 使用清晰的参数名称
   - 相关参数分组
   - 记录配置选项

3. **测试**
   - 测试训练和推理
   - 验证配置加载
   - 检查错误处理

## 4. 常见问题和解决方案

### 4.1 配置问题
- 使用前验证配置
- 提供合适的默认值
- 检查参数类型

### 4.2 内存管理
- 及时清理未使用的张量
- 使用合适的批量大小
- 监控GPU内存使用

### 4.3 训练问题
- 检查学习率设置
- 验证损失计算
- 监控梯度流动

## 5. 未来改进计划

### 5.1 短期目标
- 添加更多网络模板
- 改进错误消息
- 添加配置验证

### 5.2 长期目标
- 支持更多架构
- 添加自动化测试
- 改进文档

## 6. 使用示例

### 6.1 训练
```bash
python train.py -F "数据集路径" \
                --config config/my_network.json \
                --gpu 0 \
                --output "输出路径"
```

### 6.2 推理
```python
from predict.infer_network import infer_network

# 加载配置
with open('config/my_network.json', 'r') as f:
    config = json.load(f)
config['phase'] = 'test'

# 运行推理
output = infer_network(input_path, save_dir, config)
```
