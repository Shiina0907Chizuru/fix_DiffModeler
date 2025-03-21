#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import mrcfile
import argparse
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import re
import csv
import datetime
import traceback
# 导入pdb2vol函数，这是generate_backbone_map.py使用的核心函数

# 添加ops目录到Python路径，以便导入pdb2vol模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # 添加当前目录
from ops.pdb2vol import pdb2vol

def load_pdb(pdb_file):
    """
    加载PDB文件并提取原子坐标
    
    参数:
    - pdb_file: PDB文件路径
    
    返回:
    - atom_coords: 所有原子的坐标数组，shape为(n_atoms, 3)
    """
    # 定义骨架原子列表 - 只加载骨架原子
    backbone_list = ["CA", "C", "N"]
    
    atom_coords = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # 获取原子名称
                    atom_name = atom.get_name()
                    
                    # 只处理骨架原子
                    if atom_name in backbone_list:
                        atom_coords.append(atom.get_coord())
    
    print(f"从PDB文件中提取了{len(atom_coords)}个骨架原子(CA, C, N)")
    return np.array(atom_coords)

def load_and_threshold_density(mrc_file, contour_level):
    """
    加载密度图并应用阈值
    
    参数:
    - mrc_file: 密度图文件路径
    - contour_level: 密度图阈值
    
    返回:
    - data: 阈值处理后的密度图数据
    - voxel_size: 体素尺寸 (Å)
    - origin: 密度图原点坐标
    """
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        # 复制数据以避免修改原始文件
        data = mrc.data.copy()
        original_shape = data.shape
        print(f"密度图形状: {original_shape}")
        
        # 获取体素尺寸 (Å)
        voxel_size = np.array([
            mrc.voxel_size.x,
            mrc.voxel_size.y,
            mrc.voxel_size.z
        ])
        
        # 获取密度图原点 - 直接使用MRC文件中的原点，不做调整
        # 这与pdb2vol.py中的处理方式完全一致
        origin = np.array([
            mrc.header.origin.x,
            mrc.header.origin.y,
            mrc.header.origin.z
        ])
        print(f"使用MRC文件中的原始原点: {origin}")
        
        # 获取坐标轴排列顺序
        mapc, mapr, maps = mrc.header.mapc, mrc.header.mapr, mrc.header.maps
        axis_order = np.array([mapc, mapr, maps]) - 1  # 转为0-indexed
        
        # 将低于contour_level的值置零
        data[data < contour_level] = 0
        
        # 应用轴排列顺序
        if not np.array_equal(axis_order, np.array([0, 1, 2])):
            print(f"调整轴排列顺序: {axis_order}")
            data_reordered = np.transpose(data, np.argsort(axis_order))
        else:
            data_reordered = data
        
    return data_reordered, voxel_size, origin

def get_grid_coordinates(shape, voxel_size, origin):
    """
    计算密度图中每个体素对应的世界坐标
    
    参数:
    - shape: 密度图形状，(z_size, y_size, x_size)
    - voxel_size: 体素尺寸，(x_size, y_size, z_size)
    - origin: 密度图原点，(x_origin, y_origin, z_origin)
    
    返回:
    - x, y, z: 每个体素中心的世界坐标
    """
    # 创建体素索引网格
    i_range = np.arange(shape[0])
    j_range = np.arange(shape[1])
    k_range = np.arange(shape[2])
    
    # 使用MRC格式的顺序创建网格
    zi, yi, xi = np.meshgrid(i_range, j_range, k_range, indexing='ij')
    
    # 使用pdb2vol.py中的坐标转换逻辑
    # 从体素索引计算实际坐标
    # 公式: coord = index * voxel_size + origin
    x = xi * voxel_size[0] + origin[0]
    y = yi * voxel_size[1] + origin[1]
    z = zi * voxel_size[2] + origin[2]
    
    return x, y, z

def calculate_pdb_coverage(pdb_file, mrc_file, contour_level, distance_threshold=4.0):
    """
    计算PDB在密度图中的体素占比
    
    参数:
    - pdb_file: PDB文件路径
    - mrc_file: 密度图文件路径
    - contour_level: 密度图阈值
    - distance_threshold: 距离阈值 (Å)，默认为4.0Å
    
    返回:
    - coverage: PDB占有体素在非零体素中的占比
    - pdb_in_density_ratio: PDB占有体素在密度图中的占比
    - pdb_voxel_count: PDB占有的体素数量
    - nonzero_voxel_count: 非零体素总数
    - pdb_in_density_count: PDB区域内且密度非零的体素数量
    """
    print(f"加载PDB文件: {pdb_file}")
    atom_coords = load_pdb(pdb_file)
    print(f"PDB中原子数量: {len(atom_coords)}")
    
    print(f"加载并处理密度图: {mrc_file} (contour level: {contour_level})")
    density_data, voxel_size, origin = load_and_threshold_density(mrc_file, contour_level)
    print(f"密度图处理完成，非零体素数量: {np.sum(density_data > 0)}")
    
    # 获取所有体素的实际坐标
    print("计算体素坐标...")
    x, y, z = get_grid_coordinates(density_data.shape, voxel_size, origin)
    grid_coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    # 创建KD树，用于高效计算原子和体素之间的距离
    print("构建KD树...")
    atom_tree = cKDTree(atom_coords)
    
    # 获取密度图的线性索引
    linear_indices = np.arange(density_data.size)
    
    # 对于每个体素，找到距离最近的原子
    print(f"计算距离原子{distance_threshold}Å内的体素...")
    distances, _ = atom_tree.query(grid_coords, distance_upper_bound=distance_threshold)
    
    # 找出距离小于阈值的体素（即PDB占有的体素）
    pdb_voxels = linear_indices[distances < distance_threshold]
    
    # 找出非零密度体素
    nonzero_voxels = linear_indices[density_data.ravel() > 0]
    
    # 计算PDB占有体素在非零体素中的占比
    pdb_voxel_count = len(pdb_voxels)
    nonzero_voxel_count = len(nonzero_voxels)
    
    # 找出同时满足两个条件的体素数：距离PDB近且密度非零
    pdb_in_density_voxels = np.intersect1d(pdb_voxels, nonzero_voxels)
    pdb_in_density_count = len(pdb_in_density_voxels)
    
    # 计算占比
    pdb_coverage = pdb_in_density_count / nonzero_voxel_count if nonzero_voxel_count > 0 else 0
    pdb_in_density_ratio = pdb_in_density_count / pdb_voxel_count if pdb_voxel_count > 0 else 0
    
    return pdb_coverage, pdb_in_density_ratio, pdb_voxel_count, nonzero_voxel_count, pdb_in_density_count

def save_marked_density(pdb_file, mrc_file, output_mrc, contour_level, distance_threshold=4.0, ref_map=None):
    """
    将PDB占有的体素在密度图中标记出来并保存为新的MRC文件
    
    参数:
    - pdb_file: PDB文件路径
    - mrc_file: 密度图文件路径
    - output_mrc: 输出MRC文件路径
    - contour_level: 密度图阈值
    - distance_threshold: 距离阈值 (Å)，默认为4.0Å
    - ref_map: 参考密度图路径，用于确保坐标系统一致性（与生成的backbone_map使用的参考图相同）
    """
    print(f"加载PDB文件: {pdb_file}")
    atom_coords = load_pdb(pdb_file)
    print(f"PDB中原子数量: {len(atom_coords)}")
    
    # 读取原始MRC文件以保留其标头信息
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        # 制作数据副本
        original_data = mrc.data.copy()
        header = mrc.header
        voxel_size = mrc.voxel_size
        
        # 应用contour_level阈值
        thresholded_data = original_data.copy()
        thresholded_data[thresholded_data < contour_level] = 0
        
        # 获取体素尺寸和原点
        voxel_size_array = np.array([
            voxel_size.x,
            voxel_size.y,
            voxel_size.z
        ])
        
        # 直接使用MRC文件中的原点，不做任何调整
        origin = np.array([
            header.origin.x,
            header.origin.y,
            header.origin.z
        ])
        
        # 获取坐标轴排列顺序
        mapc, mapr, maps = header.mapc, header.mapr, header.maps
        axis_order = np.array([mapc, mapr, maps]) - 1  # 转为0-indexed
        print(f"坐标轴排列顺序: {axis_order}")
        
        # 如果指定了参考图，则使用其坐标系统
        if ref_map:
            print(f"使用参考密度图: {ref_map}")
            with mrcfile.open(ref_map, permissive=True) as ref_mrc:
                ref_header = ref_mrc.header
                ref_voxel_size = ref_mrc.voxel_size
                ref_shape = ref_mrc.data.shape
                
                print(f"参考密度图尺寸: {ref_shape}")
                print(f"参考密度图体素大小: {ref_voxel_size}")
                
                # 使用参考图的原点
                origin = np.array([
                    ref_header.origin.x,
                    ref_header.origin.y,
                    ref_header.origin.z
                ])
                print(f"参考密度图原点: {origin}")
                
                # 使用参考图的体素尺寸
                voxel_size_array = np.array([
                    ref_voxel_size.x,
                    ref_voxel_size.y,
                    ref_voxel_size.z
                ])
                
                # 使用参考图的形状
                data_shape = ref_shape
                
                # 调整原始数据的大小以匹配参考图，如果需要
                if original_data.shape != ref_shape:
                    print(f"调整密度图大小以匹配参考图: {original_data.shape} -> {ref_shape}")
                    # 重新加载原始数据，按照参考图的尺寸进行裁剪或扩展
                    with mrcfile.open(mrc_file, permissive=True) as orig_mrc:
                        data = orig_mrc.data
                        # 取交集大小
                        min_shape = [min(s1, s2) for s1, s2 in zip(data.shape, ref_shape)]
                        thresholded_data = np.zeros(ref_shape)
                        # 只复制重叠部分
                        slices = tuple(slice(0, m) for m in min_shape)
                        thresholded_data[slices] = data[slices].copy()
                        thresholded_data[thresholded_data < contour_level] = 0
        else:
            # 使用原始密度图的形状
            data_shape = original_data.shape
        
        # 创建KD树，用于高效计算原子和体素之间的距离
        atom_tree = cKDTree(atom_coords)
        
        # 获取所有体素的实际坐标，使用pdb2vol.py中的坐标转换逻辑
        x, y, z = get_grid_coordinates(data_shape, voxel_size_array, origin)
        
        # 将坐标变成列表形式，以便传递给KD树
        grid_coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        
        # 计算每个体素到最近原子的距离
        distances, _ = atom_tree.query(grid_coords, distance_upper_bound=distance_threshold)
        
        # 创建标记图
        marked_data = np.zeros(data_shape)
        marked_data.ravel()[distances < distance_threshold] = 1
        
        # 创建同时满足条件的图（既在PDB 4A内，又高于contour_level）
        combined_data = np.zeros_like(marked_data)
        combined_data = np.where(
            (marked_data > 0) & (thresholded_data > 0),
            thresholded_data,
            0
        )
        
        # 保存为新的MRC文件
        with mrcfile.new(output_mrc, overwrite=True) as new_mrc:
            new_mrc.set_data(combined_data)
            
            # 完全复制原始MRC文件的所有头信息，确保坐标系统完全一致
            new_mrc.header.mapc = header.mapc
            new_mrc.header.mapr = header.mapr
            new_mrc.header.maps = header.maps
            
            # 使用原始MRC文件的原点
            new_mrc.header.origin.x = header.origin.x
            new_mrc.header.origin.y = header.origin.y
            new_mrc.header.origin.z = header.origin.z
            
            # 复制nxstart/nystart/nzstart值，不做修改
            new_mrc.header.nxstart = header.nxstart
            new_mrc.header.nystart = header.nystart
            new_mrc.header.nzstart = header.nzstart
            
            # 设置体素尺寸 - 与原始MRC文件完全一致
            try:
                new_mrc.set_voxel_size(voxel_size.x, voxel_size.y, voxel_size.z)
            except (AttributeError, TypeError):
                try:
                    new_mrc.voxel_size = voxel_size
                except:
                    print("警告：无法设置体素尺寸，输出的MRC文件可能没有正确的体素尺寸信息")
            
            # 更新其他统计信息，但不修改坐标相关信息
            new_mrc.update_header_stats()
            
    print(f"标记的密度图已保存至: {output_mrc}")

def generate_backbone_density(pdb_file, output_mrc, ref_map=None, resolution=3.0):
    """
    生成仅包含骨架原子的密度图，使用与generate_backbone_map.py完全相同的流程
    
    参数:
    - pdb_file: PDB文件路径
    - output_mrc: 输出MRC文件路径
    - ref_map: 参考密度图路径，用于确保坐标系统一致
    - resolution: 密度图分辨率，默认为3.0
    """
    print(f"使用pdb2vol生成骨架密度图...")
    # 与generate_backbone_map.py完全相同的流程
    pdb2vol(
        input_pdb=pdb_file,
        resolution=resolution,
        output_mrc=output_mrc,
        normalize=True,
        backbone_only=True,  # 只使用骨架原子
        contour=None,  # 设置为None，避免执行contour操作
        ref_map=ref_map  # 使用参考密度图保持尺寸一致
    )
    print(f"骨架密度图生成完成: {output_mrc}")

def parse_protein_info_file(file_path):
    """
    解析包含蛋白质名称和contour值的信息文件
    
    参数:
    - file_path: 信息文件路径
    
    返回:
    - protein_info: 蛋白质信息列表，每个元素为(蛋白质名称, contour值)
    """
    protein_info = []
    
    try:
        # 尝试不同的编码方式打开文件
        encodings = ['utf-8', 'gbk', 'latin-1']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_content = f.readlines()
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码读取失败，尝试下一种编码")
                continue
            except Exception as e:
                print(f"读取文件时出错: {str(e)}")
                return []
        
        if file_content is None:
            print("所有编码方式都无法读取文件")
            return []
        
        for line_num, line in enumerate(file_content, 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            try:
                # 尝试解析不同格式
                if ',' in line:
                    # 逗号分隔格式: protein_name, contour_value
                    parts = line.split(',')
                    if len(parts) >= 2:
                        protein_name = parts[0].strip()
                        contour_value = float(parts[1].strip())
                        protein_info.append((protein_name, contour_value))
                        print(f"解析到蛋白质: {protein_name}, contour值: {contour_value}")
                    else:
                        print(f"警告: 行 {line_num} 格式错误: {line}")
                else:
                    # 空格分隔格式: protein_name contour_value
                    parts = line.split()
                    if len(parts) >= 2:
                        protein_name = parts[0].strip()
                        contour_value = float(parts[1].strip())
                        protein_info.append((protein_name, contour_value))
                        print(f"解析到蛋白质: {protein_name}, contour值: {contour_value}")
                    else:
                        print(f"警告: 行 {line_num} 格式错误: {line}")
            except ValueError as e:
                print(f"警告: 解析行 {line_num} 时出错: {str(e)}")
    except Exception as e:
        print(f"读取信息文件时出错: {str(e)}")
    
    return protein_info

def batch_process(info_file, base_dir, distance_threshold=4.0, coverage_threshold=0.7):
    """
    批量处理蛋白质，计算PDB体素覆盖并生成标记后的密度图
    
    参数:
    - info_file: 包含蛋白质名称和contour值的信息文件
    - base_dir: 包含所有蛋白质文件夹的根目录
    - distance_threshold: 距离阈值 (Å)，默认为4.0Å
    - coverage_threshold: 覆盖率阈值，低于此值的蛋白质将被单独记录，默认为0.7 (70%)
    """
    print(f"批量处理蛋白质，信息文件: {info_file}，根目录: {base_dir}")
    
    # 解析蛋白质信息文件
    protein_info = parse_protein_info_file(info_file)
    if not protein_info:
        print("错误: 信息文件为空或格式错误")
        return
    
    # 统计信息
    stats = []
    low_coverage_proteins = []
    
    # 确保base_dir路径格式正确
    base_dir = os.path.normpath(base_dir)
    
    # 获取base_dir中的所有目录
    try:
        all_dirs = {}
        print(f"扫描根目录: {base_dir}")
        if os.path.exists(base_dir):
            for dirname in os.listdir(base_dir):
                dir_path = os.path.join(base_dir, dirname)
                if os.path.isdir(dir_path):
                    all_dirs[dirname.lower()] = dir_path
            print(f"找到 {len(all_dirs)} 个目录")
        else:
            print(f"错误: 根目录不存在: {base_dir}")
            return
    except Exception as e:
        print(f"扫描目录时出错: {str(e)}")
        return
    
    # 批量处理每个蛋白质
    for protein_name, contour_value in protein_info:
        print(f"\n处理蛋白质: {protein_name}, contour值: {contour_value}")
        
        # 提取蛋白质代码（通常是4个字符，如8hc8）
        protein_code = protein_name.lower()
        if len(protein_code) > 4:
            protein_code = protein_code[:4]
        
        print(f"蛋白质代码: {protein_code}")
        
        # 查找蛋白质目录
        protein_dir = None
        
        # 1. 尝试精确匹配 PDB-蛋白质代码-EMD-*
        pattern = f"pdb-{protein_code}-emd-"
        for dirname, path in all_dirs.items():
            if pattern in dirname:
                protein_dir = path
                print(f"找到精确匹配的目录: {protein_dir}")
                break
        
        # 2. 尝试模糊匹配任何包含蛋白质代码的目录
        if not protein_dir:
            for dirname, path in all_dirs.items():
                if protein_code in dirname:
                    protein_dir = path
                    print(f"找到包含蛋白质代码的目录: {protein_dir}")
                    break
        
        if not protein_dir:
            print(f"错误: 无法找到蛋白质{protein_name}的目录")
            print(f"可用目录: {list(all_dirs.keys())}")
            continue
        
        # 确保processed目录存在
        processed_dir = os.path.join(protein_dir, "processed")
        if not os.path.exists(processed_dir):
            print(f"错误: processed目录不存在: {processed_dir}")
            print(f"尝试创建processed目录")
            try:
                os.makedirs(processed_dir, exist_ok=True)
                print(f"成功创建processed目录: {processed_dir}")
            except Exception as e:
                print(f"创建processed目录失败: {str(e)}")
                continue

        # 查找PDB文件
        pdb_file = None
        # 尝试多种可能的PDB文件命名格式
        pdb_patterns = [
            os.path.join(protein_dir, f"PDB-{protein_code}.pdb"),
            os.path.join(protein_dir, f"PDB-{protein_name}.pdb"),
            os.path.join(protein_dir, f"{protein_code}.pdb"),
            os.path.join(protein_dir, f"{protein_name}.pdb")
        ]
        
        # 先检查指定的模式
        for pattern in pdb_patterns:
            if os.path.exists(pattern):
                pdb_file = pattern
                print(f"找到匹配的PDB文件: {pdb_file}")
                break
                
        # 如果没找到，搜索整个目录中的PDB文件
        if not pdb_file:
            try:
                dir_files = os.listdir(protein_dir)
                pdb_matches = [f for f in dir_files if f.endswith('.pdb') and (protein_code in f.lower() or protein_name.lower() in f.lower())]
                if pdb_matches:
                    pdb_file = os.path.join(protein_dir, pdb_matches[0])
                    print(f"在目录中找到包含蛋白质代码的PDB文件: {pdb_file}")
            except Exception as e:
                print(f"搜索PDB文件时出错: {str(e)}")
        
        # 如果仍未找到，尝试找出目录中唯一的PDB文件
        if not pdb_file:
            try:
                pdb_files = [f for f in os.listdir(protein_dir) if f.endswith('.pdb')]
                if len(pdb_files) == 1:
                    pdb_file = os.path.join(protein_dir, pdb_files[0])
                    print(f"使用目录中唯一的PDB文件: {pdb_file}")
                elif len(pdb_files) > 1:
                    # 如果有多个PDB文件，选择最大的那个（通常是完整结构）
                    largest_size = 0
                    largest_file = None
                    for file in pdb_files:
                        file_path = os.path.join(protein_dir, file)
                        file_size = os.path.getsize(file_path)
                        if file_size > largest_size:
                            largest_size = file_size
                            largest_file = file
                    
                    if largest_file:
                        pdb_file = os.path.join(protein_dir, largest_file)
                        print(f"使用目录中最大的PDB文件: {pdb_file} (大小: {largest_size/1024:.1f} KB)")
            except Exception as e:
                print(f"搜索PDB文件时出错: {str(e)}")
        
        if not pdb_file:
            print(f"错误: 无法找到蛋白质{protein_name}的PDB文件")
            continue
        
        # 查找参考密度图
        reference_map = None
        
        # 确保processed目录存在
        processed_dir = os.path.join(protein_dir, "processed")
        if not os.path.exists(processed_dir):
            print(f"错误: processed目录不存在: {processed_dir}")
            print(f"尝试创建processed目录")
            try:
                os.makedirs(processed_dir, exist_ok=True)
                print(f"成功创建processed目录: {processed_dir}")
            except Exception as e:
                print(f"创建processed目录失败: {str(e)}")
                continue
        
        # 1. 尝试直接使用蛋白质代码.mrc的文件名格式（优先在processed目录）
        direct_mrc_file = os.path.join(processed_dir, f"{protein_code}.mrc")
        if os.path.exists(direct_mrc_file):
            reference_map = direct_mrc_file
            print(f"找到直接匹配的参考密度图: {reference_map}")
        else:
            # 2. 如果直接匹配不成功，尝试其他模式
            ref_patterns = [
                os.path.join(processed_dir, f"{protein_code}_segment.mrc"),
                os.path.join(processed_dir, f"{protein_name}.mrc"),
                os.path.join(processed_dir, f"{protein_name}_segment.mrc")
            ]
            
            # 检查指定的模式
            for pattern in ref_patterns:
                if os.path.exists(pattern):
                    reference_map = pattern
                    print(f"找到匹配的参考密度图: {reference_map}")
                    break
            
            # 3. 如果未找到参考图，尝试在processed目录中搜索包含特定关键词的MRC文件
            if not reference_map and os.path.exists(processed_dir):
                try:
                    # 按优先级搜索不同类型的MRC文件
                    processed_files = os.listdir(processed_dir)
                    segment_files = [f for f in processed_files if f.endswith('.mrc') and 'segment' in f.lower()]
                    protein_name_files = [f for f in processed_files if f.endswith('.mrc') and (protein_code in f.lower() or protein_name.lower() in f.lower())]
                    all_mrc_files = [f for f in processed_files if f.endswith('.mrc')]
                    
                    # 按优先级选择密度图
                    if protein_name_files:
                        reference_map = os.path.join(processed_dir, protein_name_files[0])
                        print(f"使用找到的包含蛋白质名称的密度图: {reference_map}")
                    elif segment_files:
                        reference_map = os.path.join(processed_dir, segment_files[0])
                        print(f"使用找到的segment密度图: {reference_map}")
                    elif all_mrc_files:
                        # 如果有多个MRC文件，选择最大的那个（可能是完整密度图）
                        largest_size = 0
                        largest_file = None
                        for file in all_mrc_files:
                            file_path = os.path.join(processed_dir, file)
                            file_size = os.path.getsize(file_path)
                            if file_size > largest_size:
                                largest_size = file_size
                                largest_file = file
                        
                        if largest_file:
                            reference_map = os.path.join(processed_dir, largest_file)
                            print(f"使用最大的MRC文件作为参考密度图: {reference_map}")
                except Exception as e:
                    print(f"搜索MRC文件时出错: {str(e)}")
        
        if not reference_map:
            print(f"错误: 无法找到蛋白质{protein_name}的参考密度图")
            continue
        
        # 生成输出文件路径
        output_mrc = os.path.join(processed_dir, f"{protein_code}_backbone_coverage.mrc")
        
        # 执行计算并生成标记后的密度图
        try:
            print(f"开始处理蛋白质 {protein_name}...")
            
            # 1. 计算PDB覆盖率统计信息
            result = calculate_pdb_coverage(
                pdb_file, 
                reference_map, 
                contour_value,
                distance_threshold=distance_threshold
            )
            
            # 检查结果
            if result is None:
                print(f"错误: 处理蛋白质 {protein_name} 失败")
                continue
                
            pdb_coverage, pdb_in_density_ratio, pdb_voxel_count, nonzero_voxel_count, pdb_in_density_count = result
            
            print(f"蛋白质 {protein_name} 统计信息计算完成")
            print(f"PDB区域内的体素数量: {pdb_voxel_count}")
            print(f"密度图中非零体素数量: {nonzero_voxel_count}")
            print(f"PDB区域内且密度非零的体素数量: {pdb_in_density_count}")
            print(f"PDB占有体素在非零体素中的占比: {pdb_coverage:.2f}")
            print(f"非零密度在PDB区域的占比: {pdb_in_density_ratio:.2f}")
            
            # 2. 生成标记的密度图
            save_marked_density(
                pdb_file=pdb_file,
                mrc_file=reference_map,
                output_mrc=output_mrc,
                contour_level=contour_value,
                distance_threshold=distance_threshold,
                ref_map=reference_map
            )
            print(f"生成标记的密度图完成: {output_mrc}")
            
            # 记录统计信息
            stats.append({
                '蛋白质名称': protein_name,
                'Contour值': contour_value,
                'PDB区域内的体素数量': pdb_voxel_count,
                '密度图中非零体素数量': nonzero_voxel_count,
                'PDB区域内且密度非零的体素数量': pdb_in_density_count,
                'PDB占有体素在非零体素中的占比': pdb_coverage,
                '非零密度在PDB区域的占比': pdb_in_density_ratio
            })
            
            # 检查覆盖率，如果低于阈值则记录
            if pdb_coverage < coverage_threshold:
                low_coverage_proteins.append((protein_name, pdb_coverage))
                print(f"警告: 蛋白质 {protein_name} 的覆盖率 ({pdb_coverage:.2f}) 低于阈值 ({coverage_threshold:.2f})")
                
        except Exception as e:
            print(f"处理蛋白质 {protein_name} 时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    # 处理完所有蛋白质后，保存统计信息
    if stats:
        # 创建CSV文件名和低覆盖率蛋白质列表文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(os.path.dirname(info_file), f"protein_coverage_stats_{timestamp}.csv")
        low_coverage_file = os.path.join(os.path.dirname(info_file), f"low_coverage_proteins_{timestamp}.txt")
        
        try:
            # 创建CSV文件并写入表头和数据
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    '蛋白质名称', 'Contour值', 'PDB区域内的体素数量', 
                    '密度图中非零体素数量', 'PDB区域内且密度非零的体素数量',
                    'PDB占有体素在非零体素中的占比', '非零密度在PDB区域的占比'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for stat in stats:
                    writer.writerow(stat)
                    
            print(f"统计信息已保存至: {csv_file}")
            
            # 创建低覆盖率蛋白质文件并写入数据
            if low_coverage_proteins:
                with open(low_coverage_file, 'w', encoding='utf-8') as f:
                    f.write(f"# 覆盖率低于{coverage_threshold*100}%的蛋白质列表\n")
                    f.write("# 格式: 蛋白质名称 覆盖率\n\n")
                    
                    for protein_name, coverage in low_coverage_proteins:
                        f.write(f"{protein_name} {coverage:.4f}\n")
                        
                print(f"低覆盖率蛋白质名单已保存至: {low_coverage_file}")
                print(f"共有 {len(low_coverage_proteins)} 个蛋白质覆盖率低于阈值 {coverage_threshold:.2f}")
        except Exception as e:
            print(f"保存统计信息时出错: {str(e)}")
    
    print(f"\n批处理完成。共处理 {len(stats)} 个蛋白质，{len(low_coverage_proteins)} 个蛋白质覆盖率低于阈值 {coverage_threshold:.2f}")
    
def main():
    parser = argparse.ArgumentParser(description='计算PDB在密度图中的体素占比或生成骨架密度图')
    
    # 创建互斥组，单个处理和批处理不能同时使用
    mode_group = parser.add_mutually_exclusive_group()
    
    # 单个处理模式参数
    mode_group.add_argument('--pdb', type=str, help='PDB文件路径')
    parser.add_argument('--mrc', type=str, help='密度图文件路径')
    parser.add_argument('--output_mrc', type=str, help='输出MRC文件路径')
    parser.add_argument('--contour', type=float, help='密度图阈值')
    parser.add_argument('--distance', type=float, default=4.0, help='距离阈值 (Å)，默认为4.0Å')
    parser.add_argument('--ref_map', type=str, help='参考密度图路径，用于确保坐标系统一致性')
    parser.add_argument('--use_pdb2vol', action='store_true', help='使用pdb2vol函数生成密度图（推荐）')
    parser.add_argument('--resolution', type=float, default=3.0, help='pdb2vol生成密度图的分辨率，默认为3.0')
    
    # 批处理模式参数
    mode_group.add_argument('--info_file', type=str, help='蛋白质信息文件路径（格式：蛋白质名称 contour值）')
    parser.add_argument('--base_dir', type=str, help='包含所有蛋白质文件夹的根目录')
    parser.add_argument('--coverage_threshold', type=float, default=0.7, 
                      help='覆盖率阈值，低于此值的蛋白质将被记录下来，默认为0.7 (70%)')
    
    args = parser.parse_args()
    
    # 检查参数是否合法
    if args.info_file and not args.base_dir:
        parser.error("使用批处理模式时需要同时指定 --info_file 和 --base_dir")
    
    # 使用pdb2vol模式
    if args.use_pdb2vol and args.pdb and args.output_mrc:
        print("使用pdb2vol函数生成密度图...")
        generate_backbone_density(
            input_file=args.pdb,
            output_mrc=args.output_mrc,
            resolution=args.resolution,
            reference_map=args.ref_map
        )
        print(f"密度图生成完成: {args.output_mrc}")
    # 批处理模式
    elif args.info_file and args.base_dir:
        print(f"批量处理蛋白质，信息文件: {args.info_file}，根目录: {args.base_dir}")
        batch_process(
            args.info_file,
            args.base_dir,
            distance_threshold=args.distance,
            coverage_threshold=args.coverage_threshold
        )
    # 单个处理模式
    elif args.pdb:
        # 验证必需参数
        if not args.mrc:
            parser.error("需要指定参数 --mrc")
        if not args.contour:
            parser.error("需要指定参数 --contour")
            
        # 计算PDB体素覆盖
        if args.output_mrc:
            print("生成标记的密度图...")
            save_marked_density(
                args.pdb, 
                args.mrc, 
                args.output_mrc, 
                args.contour,
                args.distance,
                args.ref_map
            )
        else:
            print("计算PDB体素覆盖率...")
            calculate_pdb_coverage(
                args.pdb, 
                args.mrc, 
                args.contour, 
                args.distance
            )
    else:
        parser.error("需要指定 --pdb 参数或使用批处理模式 (--info_file 和 --base_dir)")

if __name__ == '__main__':
    main()
