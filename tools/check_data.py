import os
import numpy as np
import torch
from collections import defaultdict

def analyze_data(data_path):
    """分析数据集中的input和output文件

    Args:
        data_path (str): 数据集路径
    """
    print(f"\n分析数据集: {data_path}")
    
    # 统计信息
    stats = {
        'total_maps': 0,
        'total_pairs': 0,
        'input_shapes': defaultdict(int),
        'output_shapes': defaultdict(int),
        'input_ranges': [],
        'output_ranges': [],
        'input_types': defaultdict(int),
        'output_types': defaultdict(int)
    }
    
    # 遍历所有map目录
    for map_name in os.listdir(data_path):
        cur_dir = os.path.join(data_path, map_name)
        if not os.path.isdir(cur_dir):
            continue
            
        stats['total_maps'] += 1
        print(f"\n检查map: {map_name}")
        
        # 获取input和output文件列表
        input_files = [f for f in os.listdir(cur_dir) if "input" in f and ".npy" in f]
        output_files = [f for f in os.listdir(cur_dir) if "output" in f and ".npy" in f]
        input_files.sort()
        output_files.sort()
        
        if len(input_files) != len(output_files):
            print(f"警告: input文件数({len(input_files)})与output文件数({len(output_files)})不匹配!")
            continue
            
        stats['total_pairs'] += len(input_files)
        
        # 分析每对文件
        for input_file, output_file in zip(input_files, output_files):
            input_path = os.path.join(cur_dir, input_file)
            output_path = os.path.join(cur_dir, output_file)
            
            # 加载数据
            input_data = np.load(input_path)
            output_data = np.load(output_path)
            
            # 记录形状
            stats['input_shapes'][str(input_data.shape)] += 1
            stats['output_shapes'][str(output_data.shape)] += 1
            
            # 记录数据类型
            stats['input_types'][str(input_data.dtype)] += 1
            stats['output_types'][str(output_data.dtype)] += 1
            
            # 记录数值范围
            stats['input_ranges'].append({
                'min': float(input_data.min()),
                'max': float(input_data.max()),
                'mean': float(input_data.mean()),
                'std': float(input_data.std())
            })
            stats['output_ranges'].append({
                'min': float(output_data.min()),
                'max': float(output_data.max()),
                'mean': float(output_data.mean()),
                'std': float(output_data.std())
            })
            
            # 检查是否有NaN或inf
            if np.isnan(input_data).any():
                print(f"警告: {input_file} 包含NaN值!")
            if np.isnan(output_data).any():
                print(f"警告: {output_file} 包含NaN值!")
            if np.isinf(input_data).any():
                print(f"警告: {input_file} 包含inf值!")
            if np.isinf(output_data).any():
                print(f"警告: {output_file} 包含inf值!")

    # 打印统计信息
    print("\n=== 数据集统计信息 ===")
    print(f"总map数: {stats['total_maps']}")
    print(f"总样本对数: {stats['total_pairs']}")
    
    print("\nInput形状分布:")
    for shape, count in stats['input_shapes'].items():
        print(f"  {shape}: {count}对")
        
    print("\nOutput形状分布:")
    for shape, count in stats['output_shapes'].items():
        print(f"  {shape}: {count}对")
        
    print("\nInput数据类型分布:")
    for dtype, count in stats['input_types'].items():
        print(f"  {dtype}: {count}个文件")
        
    print("\nOutput数据类型分布:")
    for dtype, count in stats['output_types'].items():
        print(f"  {dtype}: {count}个文件")
    
    # 计算数值范围的统计信息
    if stats['input_ranges']:
        input_mins = [r['min'] for r in stats['input_ranges']]
        input_maxs = [r['max'] for r in stats['input_ranges']]
        input_means = [r['mean'] for r in stats['input_ranges']]
        input_stds = [r['std'] for r in stats['input_ranges']]
        
        print("\nInput数值范围:")
        print(f"  最小值: {min(input_mins):.6f} ~ {max(input_mins):.6f}")
        print(f"  最大值: {min(input_maxs):.6f} ~ {max(input_maxs):.6f}")
        print(f"  平均值: {min(input_means):.6f} ~ {max(input_means):.6f}")
        print(f"  标准差: {min(input_stds):.6f} ~ {max(input_stds):.6f}")
    
    if stats['output_ranges']:
        output_mins = [r['min'] for r in stats['output_ranges']]
        output_maxs = [r['max'] for r in stats['output_ranges']]
        output_means = [r['mean'] for r in stats['output_ranges']]
        output_stds = [r['std'] for r in stats['output_ranges']]
        
        print("\nOutput数值范围:")
        print(f"  最小值: {min(output_mins):.6f} ~ {max(output_mins):.6f}")
        print(f"  最大值: {min(output_maxs):.6f} ~ {max(output_maxs):.6f}")
        print(f"  平均值: {min(output_means):.6f} ~ {max(output_means):.6f}")
        print(f"  标准差: {min(output_stds):.6f} ~ {max(output_stds):.6f}")

if __name__ == "__main__":
    # 设置要分析的数据集路径
    data_paths = [
        "data/train",  # 训练集路径
        "data/test"    # 测试集路径
    ]
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            analyze_data(data_path)
        else:
            print(f"\n警告: 路径 {data_path} 不存在!")
