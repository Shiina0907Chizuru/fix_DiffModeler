import torch
import torch.nn as nn
from model.UMLP3D import UMLP3D
from model.config_networks import get_network_class

def test_umlp3d():
    """测试UMLP3D实现与配置接口的兼容性"""
    print("测试UMLP3D实现...")
    
    # 参数
    batch_size = 2
    channels = 2  # 输入通道
    box_size = 32  # 3D体积大小
    time_step = 100
    
    # 创建随机3D输入张量
    x = torch.randn(batch_size, channels, box_size, box_size, box_size)
    t = torch.randint(0, time_step, (batch_size,))
    
    # UMLP3D参数
    in_channel = 2
    out_channel = 1
    inner_channel = 32
    norm_groups = 32
    channel_mults = (1, 2, 4, 8)
    attn_res = (8)
    res_blocks = 2
    dropout = 0.1
    with_noise_level_emb = True
    
    print("1. 直接创建UMLP3D模型")
    model_direct = UMLP3D(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout=dropout,
        with_noise_level_emb=with_noise_level_emb,
        box_size=box_size
    )
    
    print("2. 通过get_network_class获取UMLP3D类")
    model_class = get_network_class("umlp3d")
    model_factory = model_class(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout=dropout,
        with_noise_level_emb=with_noise_level_emb,
        box_size=box_size
    )
    
    # 测试前向传播
    print("3. 测试前向传播...")
    with torch.no_grad():
        output_direct = model_direct(x, t)
        output_factory = model_factory(x, t)
    
    # 检查输出
    print(f"输入形状: {x.shape}")
    print(f"直接模型输出形状: {output_direct.shape}")
    print(f"工厂方法模型输出形状: {output_factory.shape}")
    
    # 验证两个模型产生相同的输出形状
    assert output_direct.shape == (batch_size, out_channel, box_size, box_size, box_size), "输出形状不匹配!"
    assert output_factory.shape == (batch_size, out_channel, box_size, box_size, box_size), "输出形状不匹配!"
    
    print("UMLP3D测试成功完成!")

if __name__ == "__main__":
    test_umlp3d()