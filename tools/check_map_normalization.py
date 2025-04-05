#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查密度图是否经过标准化的脚本
检测两种常见的标准化方式：
1. Z-Score标准化 (均值≈0，标准差≈1)
2. MinMax归一化 (值域≈[0,1])
"""

import os
import sys
import numpy as np
import mrcfile
import argparse
from tabulate import tabulate

def analyze_map_file(map_path):
    """分析密度图文件，检查是否经过标准化"""
    try:
        print(f"分析文件: {os.path.basename(map_path)}")
        
        # 打开并读取密度图数据
        with mrcfile.open(map_path, permissive=True) as mrc:
            map_data = np.array(mrc.data)
            
        # 基本统计
        min_val = np.min(map_data)
        max_val = np.max(map_data)
        mean_val = np.mean(map_data)
        std_val = np.std(map_data)
        median_val = np.median(map_data)
        
        # 负值统计
        negative_count = np.sum(map_data < 0)
        negative_percent = (negative_count / map_data.size) * 100
        
        # 直方图信息
        percentile_1 = np.percentile(map_data, 1)
        percentile_99 = np.percentile(map_data, 99)
        
        # 判断标准化类型
        is_z_score = -0.1 < mean_val < 0.1 and 0.9 < std_val < 1.1
        is_minmax = -0.01 < min_val < 0.01 and 0.99 < max_val < 1.01
        is_partially_normalized = 0 <= min_val < 0.01 and max_val <= 1.01 and max_val >= 0.95
        
        # 展示结果
        stats = [
            ["形状", map_data.shape],
            ["最小值", min_val],
            ["最大值", max_val],
            ["均值", mean_val],
            ["中位数", median_val],
            ["标准差", std_val],
            ["1%分位数", percentile_1],
            ["99%分位数", percentile_99],
            ["负值数量", negative_count],
            ["负值占比", f"{negative_percent:.3f}%"],
        ]
        
        print(tabulate(stats, tablefmt="grid"))
        
        # 判断标准化状态
        print("\n标准化状态分析:")
        if is_z_score:
            print("✅ 该密度图很可能经过Z-Score标准化 (均值≈0，标准差≈1)")
        else:
            print("❌ 该密度图未经过完整的Z-Score标准化")
            
        if is_minmax:
            print("✅ 该密度图很可能经过MinMax归一化 (值域≈[0,1])")
        elif is_partially_normalized:
            print("⚠️ 该密度图可能经过某种形式的归一化，但不是严格的MinMax归一化")
        else:
            print("❌ 该密度图未经过MinMax归一化")
        
        return {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "negative_percent": negative_percent,
            "is_z_score": is_z_score,
            "is_minmax": is_minmax,
            "is_partially_normalized": is_partially_normalized
        }
        
    except Exception as e:
        print(f"分析文件失败: {e}")
        return None

def analyze_multiple_maps(dir_path, extension='.mrc'):
    """分析目录中的多个密度图文件"""
    results = []
    
    for filename in os.listdir(dir_path):
        if filename.endswith(extension):
            file_path = os.path.join(dir_path, filename)
            result = analyze_map_file(file_path)
            if result:
                results.append((filename, result))
                print("\n" + "="*80 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='检查密度图是否标准化')
    parser.add_argument('--path', type=str, required=True, help='密度图文件路径或包含密度图的文件夹路径')
    parser.add_argument('--batch', action='store_true', help='批量处理文件夹中的所有密度图')
    parser.add_argument('--ext', type=str, default='.mrc', help='密度图文件扩展名 (默认: .mrc)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("密度图标准化检测工具")
    print("="*80)
    
    if args.batch:
        if not os.path.isdir(args.path):
            print(f"错误: 指定的路径不是一个目录: {args.path}")
            return
        
        results = analyze_multiple_maps(args.path, args.ext)
        
        if results:
            print("\n总结:")
            summary = []
            for filename, result in results:
                norm_type = "Z-Score" if result["is_z_score"] else ("MinMax" if result["is_minmax"] else "未标准化")
                summary.append([
                    filename,
                    f"{result['min']:.3f}",
                    f"{result['max']:.3f}",
                    f"{result['mean']:.3f}",
                    f"{result['std']:.3f}",
                    norm_type
                ])
            
            print(tabulate(summary, 
                           headers=["文件名", "最小值", "最大值", "均值", "标准差", "标准化类型"],
                           tablefmt="grid"))
    else:
        if not os.path.isfile(args.path):
            print(f"错误: 指定的路径不是一个文件: {args.path}")
            return
        
        analyze_map_file(args.path)

if __name__ == "__main__":
    main()
