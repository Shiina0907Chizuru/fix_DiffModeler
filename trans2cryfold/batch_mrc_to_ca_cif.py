#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import mrcfile
import glob
import re
from pathlib import Path
from scipy.spatial import cKDTree  
from scipy.spatial.distance import pdist, squareform
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量处理MRC转换为CryFold可用的CIF格式")
    parser.add_argument("--input_dir", required=True, help="包含segment.mrc文件的输入目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--threshold_file", required=True, help="阈值信息文件路径")
    parser.add_argument("--ca_count_file", help="CA原子数量信息文件路径，如果不指定，将使用阈值文件")
    parser.add_argument("--ca_count_multiplier", type=float, default=1.2, help="CA原子数量的倍数，用于计算最远点取样的点数")
    parser.add_argument("--min_distance", type=float, default=3.8, help="最小CA原子间距（埃）")
    parser.add_argument("--use_fps", action="store_true", help="是否使用最远点取样")
    parser.add_argument("--default_threshold", type=float, default=0.5, help="默认密度阈值(当阈值文件中未找到对应信息时使用)")
    parser.add_argument("--default_ca_count", type=int, default=1000, help="默认CA原子数量(当无法确定CA原子数量时使用)")
    parser.add_argument("--file_pattern", default="*_segment.mrc", help="MRC文件名匹配模式，默认为*_segment.mrc，可以根据实际文件名调整")
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
    # 初始化PDB结构生成器
    builder = StructureBuilder()
    builder.init_structure("ca_structure")
    builder.init_model(1)
    builder.init_chain("A")
    builder.init_seg(" ")
    
    # 添加CA原子
    for i, coord in enumerate(points):
        # 序列号从1开始
        res_id = (" ", i+1, " ")
        builder.init_residue("GLY", *res_id)
        builder.init_atom("CA", coord, 0.0, 1.0, " ", "CA", element="C")
    
    # 获取完整结构
    structure = builder.get_structure()
    
    # 写入CIF文件
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(path_to_save)

def create_cryfold_input(input_mrc, output_dir, threshold=0.5, min_distance=3.8, use_fps=False, n_samples=1000):
    """
    将DiffModeler生成的骨架MRC转换为CryFold第二阶段所需的CIF文件
    
    参数:
    - input_mrc: 输入MRC密度图文件
    - output_dir: 输出目录
    - threshold: 密度阈值
    - min_distance: 最小CA原子间距（埃）
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    
    返回:
    - 输出CIF文件的路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 从输入文件名构建输出文件名
    base_name = os.path.basename(input_mrc).replace('.mrc', '')
    base_name = base_name.replace('_segment', '')  # 移除可能的segment后缀
    
    output_cif = os.path.join(output_dir, f"{base_name}.cif")
    output_ca_points = os.path.join(output_dir, f"{base_name}_ca_points.txt")
    
    # 如果已经存在，则跳过
    if os.path.exists(output_cif):
        print(f"文件已存在，跳过: {output_cif}")
        return output_cif
    
    # 加载MRC文件
    try:
        density, voxel_size, origin = load_mrc(input_mrc)
    except Exception as e:
        print(f"加载MRC文件时出错: {str(e)}")
        return None
    
    # 检测CA原子位置
    print(f"使用密度阈值 {threshold} 寻找CA原子...")
    ca_coords, _ = grid_to_points(
        density, 
        threshold=threshold, 
        neighbour_distance_threshold=min_distance,
        output_dir=output_dir,
        voxel_size=voxel_size,
        global_origin=origin,
        use_fps=use_fps,
        n_samples=n_samples
    )
    
    # 如果未能检测到足够的CA原子，尝试降低阈值
    reduced_threshold = threshold
    step = 0.1
    while len(ca_coords) < 10 and reduced_threshold > 0.05:  # 设置一个最低阈值
        reduced_threshold -= step
        print(f"未检测到足够的CA原子，降低阈值至 {reduced_threshold}...")
        ca_coords, _ = grid_to_points(
            density, 
            threshold=reduced_threshold, 
            neighbour_distance_threshold=min_distance,
            output_dir=output_dir,
            voxel_size=voxel_size,
            global_origin=origin,
            use_fps=use_fps,
            n_samples=n_samples
        )
        
        # 如果连续尝试还是不行，加大降幅
        if len(ca_coords) < 10 and step < 0.3:
            step += 0.1
            
        # 如果已经找到足够的CA原子，退出循环
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

def parse_threshold_file(file_path):
    """
    解析阈值信息文件
    
    参数:
    - file_path: 阈值信息文件路径
    
    返回:
    - pdb_threshold_dict: PDB ID -> 阈值的字典
    """
    pdb_threshold_dict = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(':')
                if len(parts) == 2:
                    pdb_id = parts[0].strip()
                    threshold_value = float(parts[1].strip())
                    pdb_threshold_dict[pdb_id.lower()] = threshold_value
    except Exception as e:
        print(f"解析阈值文件时出错: {str(e)}")
    
    return pdb_threshold_dict

def parse_ca_count_file(file_path):
    """
    解析CA原子数量信息文件
    
    参数:
    - file_path: CA原子数量信息文件路径（与阈值文件相同）
    
    返回:
    - pdb_ca_count_dict: PDB ID -> CA原子数量的字典
    """
    # 实际上是同一个文件，我们假设如果数值大于1，则为CA原子数量（例如图中的2.8, 2.92, 4.0等）
    pdb_ca_count_dict = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(':')
                if len(parts) == 2:
                    pdb_id = parts[0].strip()
                    value = float(parts[1].strip())
                    
                    # 如果值大于1，假设这是CA原子数量
                    if value > 1.0:
                        pdb_ca_count_dict[pdb_id.lower()] = int(value)
    except Exception as e:
        print(f"解析CA原子数量文件时出错: {str(e)}")
    
    return pdb_ca_count_dict

def extract_pdb_id(file_path):
    """
    从文件路径中提取PDB ID
    
    参数:
    - file_path: 文件路径
    
    返回:
    - pdb_id: PDB ID
    """
    # 尝试从文件名中提取PDB ID（通常是前4-6个字符）
    file_name = os.path.basename(file_path)
    
    # 优先尝试从文件名中提取标准格式的PDB ID
    pdb_match = re.search(r'^[a-zA-Z0-9]{4,6}', file_name)
    if pdb_match:
        return pdb_match.group(0).lower()
    
    # 否则使用不带扩展名和后缀的文件名
    return file_name.split('_')[0].lower()

def batch_process(args):
    """
    批量处理MRC文件
    
    参数:
    - args: 命令行参数
    """
    # 解析阈值信息
    threshold_dict = parse_threshold_file(args.threshold_file)
    print(f"已加载 {len(threshold_dict)} 个阈值信息")
    
    # 解析CA原子数量信息
    ca_count_file = args.ca_count_file if args.ca_count_file else args.threshold_file
    ca_count_dict = parse_ca_count_file(ca_count_file)
    print(f"已从{os.path.basename(ca_count_file)}加载 {len(ca_count_dict)} 个CA原子数量信息")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在")
        return
        
    if not os.path.isdir(args.input_dir):
        print(f"错误: '{args.input_dir}' 不是一个目录")
        return
    
    # 列出目录中的所有文件进行调试
    all_files = os.listdir(args.input_dir)
    print(f"目录 '{args.input_dir}' 中的所有文件:")
    for file in all_files:
        print(f"  - {file}")
    
    # 直接在输入目录中查找所有匹配的MRC文件
    pattern_path = os.path.join(args.input_dir, args.file_pattern)
    print(f"正在查找匹配模式: {pattern_path}")
    segment_mrc_files = glob.glob(pattern_path)
    
    print(f"找到 {len(segment_mrc_files)} 个待处理的MRC文件")
    
    # 如果没有找到文件，尝试其他可能的模式
    if len(segment_mrc_files) == 0:
        alternative_patterns = ["*.mrc", "*segment*.mrc", "*_segment*.mrc", "*_segment.mrc", "*segment.mrc"]
        
        for alt_pattern in alternative_patterns:
            if alt_pattern == args.file_pattern:
                continue
                
            alt_path = os.path.join(args.input_dir, alt_pattern)
            alt_files = glob.glob(alt_path)
            if len(alt_files) > 0:
                print(f"使用替代模式 '{alt_pattern}' 找到 {len(alt_files)} 个文件")
                print(f"建议使用命令行参数: --file_pattern '{alt_pattern}'")
                # 可选的自动使用替代模式
                # segment_mrc_files = alt_files
                # break
    
    # 处理每个MRC文件
    successful_count = 0
    for mrc_file in segment_mrc_files:
        pdb_id = extract_pdb_id(mrc_file)
        print(f"\n处理文件: {mrc_file} (PDB ID: {pdb_id})")
        
        # 为每个PDB创建独立的输出目录
        pdb_output_dir = os.path.join(args.output_dir, pdb_id)
        os.makedirs(pdb_output_dir, exist_ok=True)
        
        # 获取该PDB的阈值和CA原子数量
        threshold = threshold_dict.get(pdb_id, args.default_threshold)
        ca_count = ca_count_dict.get(pdb_id, args.default_ca_count)
        
        # 计算FPS采样点数量
        n_samples = int(ca_count * args.ca_count_multiplier) if args.use_fps else args.default_ca_count
        
        print(f"使用阈值: {threshold}, CA原子数量: {ca_count}, FPS采样点数: {n_samples}")
        
        # 处理MRC文件
        try:
            output_file = create_cryfold_input(
                mrc_file,
                pdb_output_dir,
                threshold=threshold,
                min_distance=args.min_distance,
                use_fps=args.use_fps,
                n_samples=n_samples
            )
            
            if output_file:
                successful_count += 1
                print(f"成功处理: {mrc_file} -> {output_file}")
            else:
                print(f"处理失败: {mrc_file}")
                
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n批处理完成，成功处理 {successful_count}/{len(segment_mrc_files)} 个文件")

def main():
    args = parse_args()
    batch_process(args)

if __name__ == "__main__":
    main()
