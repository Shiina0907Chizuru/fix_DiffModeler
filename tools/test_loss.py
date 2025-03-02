import torch
import numpy as np

def calculate_iou_bit(pred, target, eps=1e-6):
    """
    使用位运算计算IOU（原始代码中的方式）
    pred, target: 形状相同的张量，值应该是布尔值或0/1
    """
    # 确保输入是布尔值
    pred = pred.bool()
    target = target.bool()
    
    # 使用位运算计算交集和并集
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    # 计算IOU
    iou = (intersection + eps) / (union + eps)
    
    return iou, intersection, union

def calculate_iou_float(pred, target, eps=1e-6):
    """
    使用浮点数计算IOU
    pred, target: 形状相同的张量
    """
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + eps) / (union + eps)
    
    return iou, intersection, union

def calculate_dice_loss(pred, target, eps=1e-7):
    """计算Dice Loss
    Args:
        pred: 预测值 (logits)
        target: 真实值 (0-1)
        eps: 防止除零的小数
    Returns:
        loss: Dice Loss值
        dice_score: Dice系数
        intersection: 交集
        cardinality: 并集
    """
    # 将预测值转换为概率
    input_soft = torch.sigmoid(pred)
    
    # 计算交集
    dims = tuple(range(2, len(pred.shape)))  # 对H,W维度求和
    intersection = torch.sum(input_soft * target, dim=dims)
    cardinality = torch.sum(input_soft + target, dim=dims)
    
    # 计算Dice系数
    dice_score = (2. * intersection + eps) / (cardinality + eps)
    
    # 计算平均Dice Loss
    loss = 1 - dice_score.mean()
    
    # 返回标量值
    return (
        float(loss),
        float(dice_score.mean()),
        float(intersection.sum()),
        float(cardinality.sum())
    )

def calculate_iou(pred, target):
    """
    计算IOU，模拟实际模型中的计算方式
    pred: sigmoid后的预测值
    target: 真实值
    """
    # 二值化预测值
    pred = (pred >= 0.5)
    target = (target > 0.5)
    
    # 计算交集和并集
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    # 处理特殊情况：如果并集为0，返回1
    if union == 0:
        return 1.0
    
    return (intersection / union).item()

def test_case_1():
    """简单的2x2测试用例"""
    print("\n测试用例1: 2x2矩阵")
    
    # [B, C, H, W] = [1, 1, 2, 2]
    pred = torch.tensor([[[[10., -10.],
                          [-10., 10.]]]])  # [[1, 0], [0, 1]]
    target = torch.tensor([[[[1., 0.],
                            [0., 1.]]]])
    
    print("预测值(原始logits):\n", pred.squeeze())
    print("预测值(sigmoid后):\n", torch.sigmoid(pred).squeeze())
    print("真实值:\n", target.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred) >= 0.5).bool()
    target_binary = (target > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred, target)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")

def test_case_2():
    """3D测试用例"""
    print("\n测试用例2: 3D张量")
    
    # [B, C, H, W] = [2, 1, 2, 2]
    pred = torch.tensor([
        [[[10., -10.],
          [-10., 10.]]],
        [[[10., 10.],
          [-10., -10.]]]
    ])
    target = torch.tensor([
        [[[1., 0.],
          [0., 1.]]],
        [[[1., 1.],
          [0., 0.]]]
    ])
    
    print("预测值(原始logits):\n", pred.squeeze())
    print("预测值(sigmoid后):\n", torch.sigmoid(pred).squeeze())
    print("真实值:\n", target.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred) >= 0.5).bool()
    target_binary = (target > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred, target)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")

def test_case_3():
    """多通道测试用例"""
    print("\n测试用例3: 多通道")
    
    # [B, C, H, W] = [1, 2, 2, 2]
    pred = torch.tensor([
        [[[ 10., -10.],
          [-10.,  10.]],
         [[-10.,  10.],
          [ 10., -10.]]]
    ])
    target = torch.tensor([
        [[[ 1., 0.],
          [ 0., 1.]],
         [[ 0., 1.],
          [ 1., 0.]]]
    ])
    
    print("预测值(原始logits):\n", pred.squeeze())
    print("预测值(sigmoid后):\n", torch.sigmoid(pred).squeeze())
    print("真实值:\n", target.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred) >= 0.5).bool()
    target_binary = (target > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred, target)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")

def test_case_4():
    """实际数据反推"""
    print("\n测试用例4: 实际数据反推\n")

    # 尝试1：[B, C, H, W] = [1, 1, 2, 2]
    print("尝试1: 复现 iou=0.75, loss=1.0")
    pred1 = torch.tensor([
        [[[10., -10.],
          [10., -10.]]]
    ])
    target1 = torch.tensor([
        [[[1., 0.],
          [1., 1.]]]
    ])
    
    print("预测值(原始logits):\n", pred1.squeeze())
    print("预测值(sigmoid后):\n", torch.sigmoid(pred1).squeeze())
    print("真实值:\n", target1.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred1) >= 0.5).bool()
    target_binary = (target1 > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred1, target1)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")
    
    # 尝试2：[B, C, H, W] = [1, 1, 2, 2]
    print("\n尝试2: 复现 iou=1.0, loss=1.0")
    pred2 = torch.tensor([
        [[[-10., -10.],
          [-10., -10.]]]
    ])
    target2 = torch.tensor([
        [[[0., 0.],
          [0., 0.]]]
    ])
    
    print("预测值(原始logits):\n", pred2.squeeze())
    print("预测值(sigmoid后):\n", torch.sigmoid(pred2).squeeze())
    print("真实值:\n", target2.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred2) >= 0.5).bool()
    target_binary = (target2 > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred2, target2)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")

def test_case_5():
    """不太准确的预测测试"""
    print("\n测试用例5: 不太准确的预测 (batch=4, channel=3)\n")
    
    # [B, C, H, W] = [4, 3, 2, 2]
    pred = torch.tensor([
        # Batch 1
        [[[ 1.2, -0.8],     # channel 1 - 中等准确度
          [-0.5,  1.5]],
         [[ 0.9,  0.4],     # channel 2 - 低准确度
          [-0.3,  0.2]],
         [[ 2.0, -1.8],     # channel 3 - 较高准确度
          [-1.5,  1.7]]],
        
        # Batch 2
        [[[ 0.7, -0.6],
          [-0.4,  0.8]],
         [[ 1.1,  0.3],
          [-0.7,  0.5]],
         [[ 1.8, -1.2],
          [-1.4,  1.6]]],
        
        # Batch 3
        [[[ 0.5, -0.4],
          [-0.6,  0.7]],
         [[ 0.8,  0.2],
          [-0.5,  0.4]],
         [[ 1.5, -1.0],
          [-1.2,  1.4]]],
        
        # Batch 4
        [[[ 0.3, -0.2],
          [-0.3,  0.6]],
         [[ 0.6,  0.1],
          [-0.4,  0.3]],
         [[ 1.2, -0.8],
          [-1.0,  1.2]]]
    ])
    
    target = torch.tensor([
        # Batch 1
        [[[ 1., 0.],        # channel 1
          [ 0., 1.]],
         [[ 1., 1.],        # channel 2
          [ 0., 0.]],
         [[ 1., 0.],        # channel 3
          [ 0., 1.]]],
        
        # Batch 2
        [[[ 1., 0.],
          [ 0., 1.]],
         [[ 1., 0.],
          [ 0., 1.]],
         [[ 1., 0.],
          [ 0., 1.]]],
        
        # Batch 3
        [[[ 1., 0.],
          [ 0., 1.]],
         [[ 0., 0.],
          [ 1., 1.]],
         [[ 1., 0.],
          [ 0., 1.]]],
        
        # Batch 4
        [[[ 1., 0.],
          [ 0., 1.]],
         [[ 1., 1.],
          [ 0., 0.]],
         [[ 1., 0.],
          [ 0., 1.]]]
    ])
    
    print("预测值(原始logits):\n", pred.squeeze())
    print("\n预测值(sigmoid后):\n", torch.sigmoid(pred).squeeze())
    print("\n真实值:\n", target.squeeze())
    
    print("\nIOU计算 (位运算):")
    pred_binary = (torch.sigmoid(pred) >= 0.5).bool()
    target_binary = (target > 0.5).bool()
    intersection = (pred_binary & target_binary).sum().float()
    union = (pred_binary | target_binary).sum().float()
    iou = intersection / union if union > 0 else torch.tensor(1.0)
    print(f"交集: {intersection:.6f}")
    print(f"并集: {union:.6f}")
    print(f"IOU: {iou:.6f}")
    
    print("\nDice Loss计算:")
    loss, dice, inter_dice, card = calculate_dice_loss(pred, target)
    print(f"交集: {inter_dice:.6f}")
    print(f"Cardinality: {card:.6f}")
    print(f"Dice系数: {dice:.6f}")
    print(f"Loss: {loss:.6f}")

if __name__ == "__main__":
    print("开始测试Loss和IOU计算...")
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
