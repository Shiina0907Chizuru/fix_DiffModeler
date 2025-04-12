#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对CIF文件中的原子坐标进行最远点取样（FPS）
此脚本读取由mrc_to_ca_cif.py生成的CIF文件，对其中的Cα原子进行FPS，并输出新的CIF文件
"""

import os
import argparse
import numpy as np
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="对CIF文件中的原子坐标进行最远点取样")
    parser.add_argument("--input", required=True, help="输入CIF文件路径")
    parser.add_argument("--output", required=True, help="输出CIF文件路径")
    parser.add_argument("--n_samples", type=int, default=500, help="采样点数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def farthest_point_sampling(points, n_samples, seed=42):
    """
    最远点取样算法(FPS)，用于点云下采样
    
    参数:
    - points: 输入点云 (N x 3)
    - n_samples: 采样点数量
    - seed: 随机种子
    
    返回:
    - sampled_points: 采样后的点云
    """
    np.random.seed(seed)
    
    # 如果点数少于请求的采样数，直接返回所有点
    n_points = len(points)
    if n_points <= n_samples:
        return points
        
    # 初始化距离矩阵和第一个采样点（随机选择）
    indices = np.zeros(n_samples, dtype=np.int32)
    indices[0] = np.random.randint(n_points)
    distances = np.sum((points - points[indices[0]])**2, axis=1)
    
    # 迭代选择最远点
    for i in range(1, n_samples):
        indices[i] = np.argmax(distances)
        # 更新距离
        new_distances = np.sum((points - points[indices[i]])**2, axis=1)
        distances = np.minimum(distances, new_distances)
    
    return points[indices]

def read_cif_coordinates(cif_path):
    """
    从CIF文件中读取原子坐标（简化版）
    
    参数:
    - cif_path: CIF文件路径
    
    返回:
    - coords: 原子坐标数组
    """
    coords = []
    
    with open(cif_path, 'r') as f:
        lines = f.readlines()
        
    # 跳过头部，直到找到ATOM记录
    atom_lines = [line for line in lines if line.strip().startswith('ATOM')]
    
    # 提取坐标
    for line in atom_lines:
        parts = line.strip().split()
        try:
            # CIF文件中坐标通常位于10-12位置 (x, y, z)
            x = float(parts[10])
            y = float(parts[11])
            z = float(parts[12])
            coords.append([x, y, z])
        except (IndexError, ValueError) as e:
            print(f"警告: 无法解析行 '{line.strip()}': {str(e)}")
            continue
    
    return np.array(coords)

def points_to_pdb(path_to_save, points):
    """
    将点云保存为CIF格式，使用与mrc_to_ca_cif.py相同的方法
    
    参数:
    - path_to_save: 输出文件路径
    - points: Cα原子坐标数组 (N x 3)
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(path_to_save)
    if output_dir:  # 只有当目录名非空时才创建目录
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建结构
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")  # 使用链ID "A"
    
    for i, point in enumerate(points):
        struct.set_line_counter(i)
        # 使用标准的氨基酸命名和编号，从1开始
        struct.init_residue("ALA", " ", i, " ")  # 残基编号从1开始
        # 设置原子
        struct.init_atom("CA", point, 0, 1, " ", "CA", "C")
    
    # 获取构建的结构
    structure = struct.get_structure()
    
    # 保存为CIF格式
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(path_to_save)
    
    print(f"成功保存CIF文件到: {path_to_save}")

def main():
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取CIF文件中的原子坐标
    print(f"读取CIF文件: {args.input}")
    coords = read_cif_coordinates(args.input)
    print(f"原始CIF文件包含 {len(coords)} 个原子")
    
    if len(coords) == 0:
        print("错误: 未能从CIF文件中提取到坐标")
        return
    
    # 执行最远点取样
    print(f"执行最远点取样, 目标采样数: {args.n_samples}")
    sampled_coords = farthest_point_sampling(coords, args.n_samples, args.seed)
    print(f"采样后得到 {len(sampled_coords)} 个原子")
    
    # 保存为新的CIF文件
    points_to_pdb(args.output, sampled_coords)
    
    # 保存采样点坐标（用于调试）
    debug_file = os.path.splitext(args.output)[0] + "_points.txt"
    with open(debug_file, 'w') as f:
        f.write("# x y z\n")
        for point in sampled_coords:
            f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
    print(f"采样点坐标已保存至: {debug_file}")

if __name__ == "__main__":
    main()
