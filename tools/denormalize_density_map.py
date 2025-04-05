#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密度图反归一化工具
将DiffModeler的trace_backbone输出的[-1,1]范围的密度图还原为原始值域

处理流程：
1. 反向sigmoid变换：将[-1,1]映射回[0,1]
2. 反向MinMax归一化：将[0,1]映射回原始值域
3. Z标准化：将数据标准化为均值为0，标准差为1

作者: Cascade
日期: 2025-04-02
"""

import os
import sys
import numpy as np
import mrcfile
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

def analyze_map(map_path):
    """分析密度图的基本统计信息"""
    with mrcfile.open(map_path, permissive=True) as mrc:
        map_data = np.array(mrc.data)
        
    # 基本统计
    stats = {
        "shape": map_data.shape,
        "min": np.min(map_data),
        "max": np.max(map_data),
        "mean": np.mean(map_data),
        "std": np.std(map_data),
        "median": np.median(map_data),
        "neg_count": np.sum(map_data < 0),
        "neg_percent": (np.sum(map_data < 0) / map_data.size) * 100
    }
    
    # 显示基本信息
    print("\n密度图分析结果:")
    stats_table = [
        ["形状", stats["shape"]],
        ["最小值", stats["min"]],
        ["最大值", stats["max"]],
        ["均值", stats["mean"]],
        ["标准差", stats["std"]],
        ["中位数", stats["median"]],
        ["负值数量", stats["neg_count"]],
        ["负值占比", f"{stats['neg_percent']:.2f}%"]
    ]
    print(tabulate(stats_table, tablefmt="grid"))
    
    return map_data, stats

def sigmoid_inverse(map_data):
    """
    将[-1,1]范围的数据转换回[0,1]范围
    这是对 x_predict = x_predict*2 -1 的反向操作
    """
    print("\n执行反向sigmoid变换: (data + 1) / 2")
    return (map_data + 1) / 2

def minmax_inverse(normalized_data, original_min, original_max):
    """
    反向执行MinMax归一化，将[0,1]范围数据还原到原始值域
    """
    print(f"\n执行反向MinMax归一化: data * (max - min) + min")
    print(f"原始值域: [{original_min}, {original_max}]")
    return normalized_data * (original_max - original_min) + original_min

def z_score_normalize(data):
    """
    执行Z标准化，将数据转换为均值为0，标准差为1
    """
    print("\n执行Z标准化: (data - mean) / std")
    mean = np.mean(data)
    std = np.std(data)
    print(f"均值: {mean:.6f}, 标准差: {std:.6f}")
    return (data - mean) / std

def generate_histogram(data, title, output_path=None):
    """生成数据直方图并保存"""
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=100, alpha=0.7)
    plt.title(title)
    plt.xlabel('密度值')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"直方图已保存至: {output_path}")
    else:
        plt.show()
    plt.close()

def save_mrc(data, template_path, output_path):
    """保存MRC文件，保留原始文件的头信息"""
    with mrcfile.open(template_path, permissive=True) as template:
        # 获取原始文件的头信息
        header = template.header
        voxel_size = template.voxel_size
        
    # 创建新的MRC文件
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        
        # 复制原始头信息
        mrc.header.origin = header.origin
        mrc.header.nxstart = header.nxstart
        mrc.header.nystart = header.nystart
        mrc.header.nzstart = header.nzstart
        mrc.header.mapc = header.mapc
        mrc.header.mapr = header.mapr
        mrc.header.maps = header.maps
        
        # 设置体素大小
        vsize = mrc.voxel_size
        vsize.flags.writeable = True
        vsize.x = voxel_size.x
        vsize.y = voxel_size.y
        vsize.z = voxel_size.z
        mrc.voxel_size = vsize
        
        mrc.update_header_from_data()
        mrc.update_header_stats()
    
    print(f"已保存处理结果至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='DiffModeler密度图反归一化工具')
    parser.add_argument('--traced', type=str, required=True, help='trace_backbone输出的密度图路径')
    parser.add_argument('--original', type=str, help='原始密度图路径（用于获取原始值域）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--min', type=float, help='原始密度图最小值（如果不提供原始图）')
    parser.add_argument('--max', type=float, help='原始密度图最大值（如果不提供原始图）')
    parser.add_argument('--hist', action='store_true', help='生成直方图')
    parser.add_argument('--z_norm', action='store_true', default=True, help='应用Z标准化')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DiffModeler密度图反归一化工具")
    print("="*80)
    
    # 1. 加载和分析trace_backbone输出的密度图
    print(f"\n正在处理traced_backbone密度图: {args.traced}")
    traced_data, traced_stats = analyze_map(args.traced)
    
    # 2. 反向sigmoid变换: [-1,1] -> [0,1]
    sigmoid_reverted_data = sigmoid_inverse(traced_data)
    
    # 显示变换后的统计信息
    sigmoid_stats = {
        "min": np.min(sigmoid_reverted_data),
        "max": np.max(sigmoid_reverted_data),
        "mean": np.mean(sigmoid_reverted_data),
        "std": np.std(sigmoid_reverted_data)
    }
    print("\n反向sigmoid变换后的统计信息:")
    sigmoid_table = [
        ["最小值", sigmoid_stats["min"]],
        ["最大值", sigmoid_stats["max"]],
        ["均值", sigmoid_stats["mean"]],
        ["标准差", sigmoid_stats["std"]]
    ]
    print(tabulate(sigmoid_table, tablefmt="grid"))
    
    # 3. 获取原始值域
    if args.original:
        print(f"\n正在分析原始密度图: {args.original}")
        original_data, original_stats = analyze_map(args.original)
        original_min = original_stats["min"]
        original_max = original_stats["max"]
    elif args.min is not None and args.max is not None:
        print(f"\n使用用户提供的原始值域:")
        original_min = args.min
        original_max = args.max
        print(f"最小值: {original_min}")
        print(f"最大值: {original_max}")
    else:
        print("\n未提供原始值域信息。将保持[0,1]范围。")
        original_min = 0
        original_max = 1
    
    # 4. 反向MinMax归一化: [0,1] -> [original_min, original_max]
    if original_min != 0 or original_max != 1:
        denormalized_data = minmax_inverse(sigmoid_reverted_data, original_min, original_max)
    else:
        denormalized_data = sigmoid_reverted_data
    
    # 显示反MinMax归一化后的统计信息
    denormalized_stats = {
        "min": np.min(denormalized_data),
        "max": np.max(denormalized_data),
        "mean": np.mean(denormalized_data),
        "std": np.std(denormalized_data)
    }
    print("\n反MinMax归一化后的统计信息:")
    denormalized_table = [
        ["最小值", denormalized_stats["min"]],
        ["最大值", denormalized_stats["max"]],
        ["均值", denormalized_stats["mean"]],
        ["标准差", denormalized_stats["std"]]
    ]
    print(tabulate(denormalized_table, tablefmt="grid"))
    
    # 5. 应用Z标准化 (新增步骤)
    if args.z_norm:
        final_data = z_score_normalize(denormalized_data)
    else:
        final_data = denormalized_data
    
    # 显示最终结果统计信息
    final_stats = {
        "min": np.min(final_data),
        "max": np.max(final_data),
        "mean": np.mean(final_data),
        "std": np.std(final_data)
    }
    print("\n最终处理结果统计信息:")
    final_table = [
        ["最小值", final_stats["min"]],
        ["最大值", final_stats["max"]],
        ["均值", final_stats["mean"]],
        ["标准差", final_stats["std"]]
    ]
    print(tabulate(final_table, tablefmt="grid"))
    
    # 6. 生成直方图（可选）
    if args.hist:
        print("\n生成密度分布直方图...")
        output_dir = os.path.dirname(args.output) if args.output else '.'
        
        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir) and output_dir != '.':
            os.makedirs(output_dir)
        
        # 生成不同阶段的直方图
        generate_histogram(
            traced_data, 
            "Traced Backbone输出密度分布 [-1,1]",
            os.path.join(output_dir, "traced_histogram.png")
        )
        
        generate_histogram(
            sigmoid_reverted_data, 
            "反向Sigmoid变换后的密度分布 [0,1]",
            os.path.join(output_dir, "sigmoid_reverted_histogram.png")
        )
        
        generate_histogram(
            denormalized_data, 
            "反MinMax归一化后的密度分布",
            os.path.join(output_dir, "denormalized_histogram.png")
        )
        
        generate_histogram(
            final_data, 
            "Z标准化后的密度分布",
            os.path.join(output_dir, "final_histogram.png")
        )
    
    # 7. 保存结果
    if args.output:
        save_mrc(final_data, args.traced, args.output)
    else:
        print("\n未指定输出路径，结果未保存。")

if __name__ == "__main__":
    main()
