#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import mrcfile
from scipy.spatial import cKDTree  
from scipy.spatial.distance import pdist, squareform
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将骨架MRC转换为CryFold可用的CIF格式")
    parser.add_argument("--input", required=True, help="输入MRC文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="密度阈值")
    parser.add_argument("--min_distance", type=float, default=3.8, help="最小CA原子间距（埃）")
    return parser.parse_args()

def load_mrc(mrc_path):
    """
    加载MRC文件并返回密度图、体素大小和原点
    
    参数:
    - mrc_path: MRC文件路径
    
    返回:
    - density: 密度图数组
    - voxel_size: 体素大小 (x, y, z)
    - origin: 原点坐标 (x, y, z)
    """
    with mrcfile.open(mrc_path) as mrc:
        density = mrc.data
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
        
        # 计算原点坐标（考虑起始位置）
        if np.all(origin == 0):  # 如果原点在(0,0,0)
            origin = np.array([
                mrc.header.nxstart * voxel_size[0],
                mrc.header.nystart * voxel_size[1],
                mrc.header.nzstart * voxel_size[2]
            ])
    
    print(f"MRC信息: 体素大小 = {voxel_size}, 原点 = {origin}")
    print(f"密度图形状: {density.shape}, 值范围: [{density.min():.4f}, {density.max():.4f}]")
    
    return density, voxel_size, origin

def get_lattice_meshgrid_np(shape_size, no_shift=False):
    """
    创建网格坐标
    
    参数:
    - shape_size: 网格形状
    - no_shift: 是否不进行偏移
    
    返回:
    - mesh: 网格坐标
    """
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh

def grid_to_points(grid, threshold, neighbour_distance_threshold):
    """
    从密度网格中提取点云，使用CryFold原始方法
    
    参数:
    - grid: 密度网格
    - threshold: 密度阈值
    - neighbour_distance_threshold: 邻居距离阈值
    
    返回:
    - output_points: 经过处理后的点云
    - output_points_before_pruning: 处理前的点云
    """
    # 创建网格坐标
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape, no_shift=False), -1)
    
    # 提取高于阈值的点
    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))
    
    # 如果没有点，直接返回空
    if len(output_points_before_pruning) == 0:
        print(f"警告: 未在阈值{threshold}处找到任何点! 请尝试降低阈值。")
        return np.array([]), np.array([])
    
    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]
    
    # 合并临近点
    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0
        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, 1.1)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]
    
    # 过滤孤立点
    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    
    print(f"提取了{len(output_points)}个点 (从原始的{len(output_points_before_pruning)}个点中)")
    
    return output_points, output_points_before_pruning

def points_to_pdb(path_to_save, points):
    """
    将点云保存为CIF格式，使用与CryFold完全相同的方法
    
    参数:
    - path_to_save: 输出文件路径
    - points: Cα原子坐标数组 (N x 3)
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    
    # 创建结构
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")  # 使用链ID "1"，与模范文件一致
    
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

def create_cryfold_input(input_mrc, output_dir, threshold=0.5, min_distance=3.8):
    """
    将DiffModeler生成的骨架MRC转换为CryFold第二阶段所需的CIF文件
    
    参数:
    - input_mrc: 输入MRC密度图文件
    - output_dir: 输出目录
    - threshold: 密度阈值
    - min_distance: 最小Cα原子间距（埃）
    
    返回:
    - 输出CIF文件的路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    see_alpha_dir = os.path.join(output_dir, "see_alpha_output")
    os.makedirs(see_alpha_dir, exist_ok=True)
    
    # 输出CA点云（用于调试）
    output_ca_points = os.path.join(see_alpha_dir, f"{os.path.basename(input_mrc).split('.')[0]}_points.txt")
    
    # 输出CIF文件路径
    output_cif = os.path.join(see_alpha_dir, f"{os.path.basename(input_mrc).split('.')[0]}_2cry.cif")
    
    # 加载MRC文件
    density, voxel_size, origin = load_mrc(input_mrc)
    
    # 按照CryFold方式从网格提取点云
    neighbour_distance_threshold = 6 / np.min(voxel_size)  # 与CryFold使用相同的邻居距离计算
    ca_coords, ca_coords_before_pruning = grid_to_points(density, threshold, neighbour_distance_threshold)
    
    # 如果没有检测到任何点，尝试降低阈值
    if len(ca_coords) == 0:
        print("警告: 未检测到任何CA原子! 尝试降低阈值...")
        for reduced_threshold in [0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
            print(f"尝试阈值: {reduced_threshold}")
            ca_coords, ca_coords_before_pruning = grid_to_points(density, reduced_threshold, neighbour_distance_threshold)
            if len(ca_coords) > 0:
                print(f"使用阈值 {reduced_threshold} 成功检测到 {len(ca_coords)} 个CA原子")
                break
    
    # 将体素坐标转换为真实坐标
    real_coords = ca_coords * voxel_size[None] + origin[None]
    
    # 保存原始点云（用于调试）
    with open(output_ca_points, 'w') as f:
        f.write("# x y z\n")
        for point in real_coords:
            f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
    
    # 格式化坐标，确保三位小数精度
    formatted_coords = np.array([[float(f"{p:.3f}") for p in point] for point in real_coords])
    
    # 将点云转换为CIF文件
    points_to_pdb(output_cif, formatted_coords)
    
    # 输出信息
    print(f"已检测到 {len(ca_coords)} 个Cα原子位置")
    print(f"已将结果保存至: {output_cif}")
    print(f"现在您可以使用该目录运行CryFold的第二阶段：")
    print(f"  cd /path/to/CryFold")
    print(f"  conda activate CryFold")
    print(f"  python -m CryFold.CryNet.inference --fasta <序列文件> --struct {output_cif} --map-path <密度图文件> --output-dir <输出目录>")
    
    return output_cif

def main():
    args = parse_args()
    create_cryfold_input(
        args.input, 
        args.output_dir, 
        args.threshold, 
        args.min_distance
    )

if __name__ == "__main__":
    main()
