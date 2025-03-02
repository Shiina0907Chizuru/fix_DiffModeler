import torch
import numpy as np

def iou(pred, target, eps=1e-6):
    """计算IOU
    Args:
        pred: 预测值 (binary)
        target: 目标值 (binary)
        eps: 数值稳定性的小值
    """
    intersection = (pred & target).float().sum(dim=(2, 3, 4))  # sum along H, W, L dimensions
    union = (pred | target).float().sum(dim=(2, 3, 4))  # sum along H, W, L dimensions
    iou = (intersection + eps) / (union + eps)
    return iou.mean(dim=0)  # return mean IoU across batch dimension

def test_special_case():
    """测试特殊情况：全0预测和全0目标"""
    print("\n测试全0情况:")
    pred = torch.zeros((2, 1, 64, 64, 64))  # [B, C, H, W, L]
    target = torch.zeros((2, 1, 64, 64, 64))
    
    iou_val = iou(pred.bool(), target.bool())
    print(f"全0 IOU: {iou_val.item()}")
    
    # 计算Dice Loss: 2|X∩Y|/(|X|+|Y|)
    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    cardinality = torch.sum(pred + target, dim=(2, 3, 4))
    dice = (2. * intersection) / (cardinality + 1e-6)
    loss = 1 - dice.mean()
    print(f"全0 Loss: {loss.item()}\n")

def test_edge_cases():
    """测试各种边缘情况"""
    # 测试用例1：预测值接近0，目标也是0
    print("测试用例1: 预测值接近0，目标为0")
    pred = torch.full((2, 1, 64, 64, 64), 0.01)
    target = torch.zeros((2, 1, 64, 64, 64))
    pred_binary = (pred >= 0.5)
    iou_val = iou(pred_binary, target.bool())
    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    cardinality = torch.sum(pred + target, dim=(2, 3, 4))
    dice = (2. * intersection) / (cardinality + 1e-6)
    loss = 1 - dice.mean()
    print(f"IOU: {iou_val.item()}")
    print(f"Loss: {loss.item()}\n")

    # 测试用例2：预测值为1，目标为0
    print("测试用例2: 预测值为1，目标为0")
    pred = torch.ones((2, 1, 64, 64, 64))
    target = torch.zeros((2, 1, 64, 64, 64))
    pred_binary = (pred >= 0.5)
    iou_val = iou(pred_binary, target.bool())
    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    cardinality = torch.sum(pred + target, dim=(2, 3, 4))
    dice = (2. * intersection) / (cardinality + 1e-6)
    loss = 1 - dice.mean()
    print(f"IOU: {iou_val.item()}")
    print(f"Loss: {loss.item()}\n")

    # 测试用例3：sigmoid输出的边缘值
    print("测试用例3: sigmoid输出的边缘值 (0.75)")
    pred = torch.full((2, 1, 64, 64, 64), 0.75)
    target = torch.zeros((2, 1, 64, 64, 64))
    pred_binary = (pred >= 0.5)
    iou_val = iou(pred_binary, target.bool())
    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    cardinality = torch.sum(pred + target, dim=(2, 3, 4))
    dice = (2. * intersection) / (cardinality + 1e-6)
    loss = 1 - dice.mean()
    print(f"IOU: {iou_val.item()}")
    print(f"Loss: {loss.item()}\n")

def verify_actual_values(logits_value):
    """验证实际的logits值"""
    print(f"\n验证logits值 = {logits_value}:")
    # 创建tensor
    pred_logits = torch.full((2, 1, 64, 64, 64), logits_value)
    target = torch.zeros((2, 1, 64, 64, 64))
    
    # 应用sigmoid
    pred_probs = torch.sigmoid(pred_logits)
    print(f"Sigmoid后的预测值: {pred_probs[0,0,0,0,0].item()}")
    
    # 计算二值预测
    pred_binary = (pred_probs >= 0.5)
    
    # 计算IOU
    iou_val = iou(pred_binary, target.bool())
    print(f"IOU: {iou_val.item()}")
    
    # 计算Dice Loss
    intersection = torch.sum(pred_probs * target, dim=(2, 3, 4))
    cardinality = torch.sum(pred_probs + target, dim=(2, 3, 4))
    dice = (2. * intersection) / (cardinality + 1e-6)
    loss = 1 - dice.mean()
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    print("验证IOU和Loss的对应关系...")
    test_special_case()
    test_edge_cases()
    
    # 验证实际观察到的值
    verify_actual_values(0.0134)  # 从日志中看到的gamma_t值
