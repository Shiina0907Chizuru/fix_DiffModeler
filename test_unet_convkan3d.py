import torch
from model.UNet3D import UNet_ConvKan3D
from model.config_networks import get_network_class

def test_unet_convkan3d():
    print("=== 测试 UNet_ConvKan3D 模型 ===")
    
    # 直接创建模型
    model = UNet_ConvKan3D(
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=[8],  # 改成列表
        res_blocks=2,
        dropout=0,
        with_noise_level_emb=True,
        box_size=32,
    )
    print(f"模型已创建 - 参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    x = torch.randn(2, 2, 32, 32, 32)  # [B, C, D, H, W]
    time = torch.ones(2, 1)
    
    try:
        with torch.no_grad():
            y = model(x, time)
        print(f"前向传播成功 - 输出形状: {y.shape}")
    except Exception as e:
        print(f"前向传播出错: {e}")
    
    # 测试通过 config_networks 获取
    print("\n=== 测试 config_networks.get_network_class ===")
    try:
        NetworkClass = get_network_class('unet_convkan3d')
        print(f"获取网络类成功: {NetworkClass.__name__}")
        
        model2 = NetworkClass(
            in_channel=2,
            out_channel=1,
            inner_channel=32,
            attn_res=[8],  # 改成列表
            box_size=16,
        )
        print(f"通过 get_network_class 创建模型成功")
    except Exception as e:
        print(f"通过 get_network_class 创建模型失败: {e}")

if __name__ == "__main__":
    test_unet_convkan3d()
