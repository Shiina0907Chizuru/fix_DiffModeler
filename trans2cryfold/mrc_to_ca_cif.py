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
    parser.add_argument("--use_fps", action="store_true", help="是否使用最远点取样")
    parser.add_argument("--n_samples", type=int, default=1000, help="最远点取样的采样点数量")
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

def farthest_point_sampling(points, n_samples):
    """
    最远点取样算法(FPS)，用于点云下采样
    
    参数:
    - points: 输入点云 (N x 3)
    - n_samples: 采样点数量
    
    返回:
    - sampled_points: 采样后的点云 (n_samples x 3)
    """
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

def grid_to_points(grid, threshold=0.5, neighbour_distance_threshold=3.8, output_dir=None, voxel_size=None, global_origin=None, use_fps=False, n_samples=1000):
    """
    将密度网格转换为点云，使用CryFold中的均值漂移方法处理
    
    参数:
    - grid: 密度网格
    - threshold: 密度阈值
    - neighbour_distance_threshold: 邻居距离阈值
    - output_dir: 输出目录，用于保存点集密度图
    - voxel_size: 体素大小，用于保存密度图
    - global_origin: 全局原点，用于保存密度图
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    
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
    
    # 应用最远点取样
    if use_fps and len(output_points) > n_samples:
        points_before_fps = len(output_points)
        output_points = farthest_point_sampling(output_points, n_samples)
        print(f"应用最远点取样后，点数从{points_before_fps}减少到{len(output_points)}")
    
    print(f"提取了{len(output_points)}个点 (从原始的{len(output_points_before_pruning)}个点中)")
    
    # 将点集转换为密度图并保存
    if output_dir is not None and voxel_size is not None and global_origin is not None:
        try:
            # 点集转密度图函数
            def points_to_grid(points, shape):
                """将点集转换为密度图"""
                density_grid = np.zeros(shape, dtype=np.float32)
                for point in points:
                    i, j, k = np.round(point).astype(int)
                    if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                        density_grid[i, j, k] = 1.0
                return density_grid
                
            # 保存处理前的点集密度图
            if len(output_points_before_pruning) > 0:
                before_pruning_grid = points_to_grid(output_points_before_pruning, grid.shape)
                before_pruning_path = os.path.join(output_dir, "points_before_pruning.mrc")
                save_dens_map(before_pruning_path, before_pruning_grid, voxel_size, global_origin)
                print(f"已保存处理前点集密度图: {before_pruning_path}")
            
            # 保存处理后的点集密度图
            if len(output_points) > 0:
                after_pruning_grid = points_to_grid(output_points, grid.shape)
                after_pruning_path = os.path.join(output_dir, "points_after_pruning.mrc")
                save_dens_map(after_pruning_path, after_pruning_grid, voxel_size, global_origin)
                print(f"已保存处理后点集密度图: {after_pruning_path}")
                
        except Exception as e:
            print(f"保存点集密度图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
        struct.init_residue(f"ALA", " ", i, " ")  # 残基编号从1开始
        # 设置原子
        struct.init_atom("CA", point, 0, 1, " ", "CA", "C")
    
    # 获取构建的结构
    structure = struct.get_structure()
    
    # 保存为CIF格式
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(path_to_save)
    
    print(f"成功保存CIF文件到: {path_to_save}")

def create_cryfold_input(input_mrc, output_dir, threshold=0.5, min_distance=3.8, use_fps=False, n_samples=1000):
    """
    将DiffModeler生成的骨架MRC转换为CryFold第二阶段所需的CIF文件
    
    参数:
    - input_mrc: 输入MRC密度图文件
    - output_dir: 输出目录
    - threshold: 密度阈值
    - min_distance: 最小Cα原子间距（埃）
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    
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
    ca_coords, ca_coords_before_pruning = grid_to_points(density, threshold, neighbour_distance_threshold, see_alpha_dir, voxel_size, origin, use_fps, n_samples)
    
    # 如果没有检测到任何点，尝试降低阈值
    if len(ca_coords) == 0:
        print("警告: 未检测到任何CA原子! 尝试降低阈值...")
        for reduced_threshold in [0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
            print(f"尝试阈值: {reduced_threshold}")
            ca_coords, ca_coords_before_pruning = grid_to_points(density, reduced_threshold, neighbour_distance_threshold, see_alpha_dir, voxel_size, origin, use_fps, n_samples)
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

def save_dens_map(save_map_path, new_dens, current_voxel_size, current_origin):
    """
    保存密度图为MRC文件，使用当前的体素大小和原点
    
    参数：
    save_map_path: 保存路径
    new_dens: 新的密度图数据（预测结果）
    current_voxel_size: 当前体素大小
    current_origin: 当前全局原点
    """
    data_new = np.float32(new_dens)
    
    # 创建新的MRC文件
    with mrcfile.new(save_map_path, overwrite=True) as mrc_new:
        # 设置数据
        mrc_new.set_data(data_new)
        
        # 设置体素大小
        vsize = mrc_new.voxel_size
        vsize.flags.writeable = True
        
        # 确保体素大小是float类型
        if isinstance(current_voxel_size, np.ndarray):
            if current_voxel_size.size >= 3:
                vsize.x = float(current_voxel_size[0])
                vsize.y = float(current_voxel_size[1])
                vsize.z = float(current_voxel_size[2])
            elif current_voxel_size.size == 1:
                vs_value = float(current_voxel_size.item())
                vsize.x = vs_value
                vsize.y = vs_value
                vsize.z = vs_value
        else:
            vsize.x = float(current_voxel_size)
            vsize.y = float(current_voxel_size)
            vsize.z = float(current_voxel_size)
        
        mrc_new.voxel_size = vsize
        
        # 设置原点
        if isinstance(current_origin, np.ndarray):
            if current_origin.size >= 3:
                mrc_new.header.origin.x = float(current_origin[0])
                mrc_new.header.origin.y = float(current_origin[1])
                mrc_new.header.origin.z = float(current_origin[2])
            elif current_origin.size == 1:
                orig_value = float(current_origin.item())
                mrc_new.header.origin.x = orig_value
                mrc_new.header.origin.y = orig_value
                mrc_new.header.origin.z = orig_value
        else:
            mrc_new.header.origin.x = float(current_origin)
            mrc_new.header.origin.y = float(current_origin)
            mrc_new.header.origin.z = float(current_origin)
        
        # 设置标准轴排列
        mrc_new.header.mapc = 1  # 按照标准约定: mapc=1, mapr=2, maps=3
        mrc_new.header.mapr = 2
        mrc_new.header.maps = 3
        
        # 更新头部统计信息
        mrc_new.update_header_stats()

def main():
    args = parse_args()
    create_cryfold_input(
        args.input, 
        args.output_dir, 
        args.threshold, 
        args.min_distance,
        args.use_fps,
        args.n_samples
    )

if __name__ == "__main__":
    main()
