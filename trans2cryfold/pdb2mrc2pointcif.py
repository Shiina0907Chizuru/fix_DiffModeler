#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批处理将PDB文件转换为MRC，再将MRC转换为CryFold兼容的CIF文件
"""

import os
import argparse
import numpy as np
import sys
import re
import mrcfile
from pathlib import Path
import tempfile
import shutil
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO
from scipy.ndimage import fourier_gaussian, gaussian_filter, zoom
from scipy.fftpack import fftn, ifftn

# 添加项目根目录到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 原子质量字典
atom_mass_dict = {
    "H": 1.008,
    "C": 12.011,
    "CA": 12.011,  # for PDB files without element notations
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.066,
}

# 从PDB或CIF文件获取原子列表
def get_atom_list(pdb_file, backbone_only=False):
    """
    获取PDB或CIF文件中的原子坐标和类型
    
    参数:
    - pdb_file: PDB或CIF文件路径
    - backbone_only: 是否只返回骨架原子
    
    返回:
    - atom_list: 原子坐标数组
    - atom_type_list: 原子类型列表
    """
    # 使用小写扩展名进行检查，确保大小写不敏感
    file_ext = os.path.splitext(pdb_file)[1].lower()
    
    try:
        # 首先尝试使用BioPython的解析器处理文件
        try:
            if file_ext == ".pdb":
                st_parser = PDBParser(QUIET=True)
                print(f"使用PDBParser解析文件: {pdb_file}")
            elif file_ext == ".cif":
                st_parser = MMCIFParser(QUIET=True)
                print(f"使用MMCIFParser解析文件: {pdb_file}")
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}，只支持.pdb和.cif格式")
                
            structure = st_parser.get_structure("protein", pdb_file)
            atom_list = []
            atom_type_list = []

            if backbone_only:
                print("只提取骨架原子...")
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            try:
                                if "CA" in residue:
                                    atom_list.append(residue["CA"].get_coord())
                                    atom_type_list.append(residue["CA"].element)
                                    if "C" in residue:
                                        atom_list.append(residue["C"].get_coord())
                                        atom_type_list.append(residue["C"].element)
                                    if "N" in residue:
                                        atom_list.append(residue["N"].get_coord())
                                        atom_type_list.append(residue["N"].element)
                            except Exception as e:
                                print(f"处理残基时出错，跳过: {e}")
                                continue
            else:
                print("提取所有原子...")
                for atom in structure.get_atoms():
                    try:
                        atom_list.append(atom.get_coord())
                        atom_type_list.append(atom.element)
                    except Exception as e:
                        print(f"处理原子时出错，跳过: {e}")
                        continue
                    
            print(f"提取了{len(atom_list)}个原子")
            
            if len(atom_list) == 0:
                raise ValueError("未能从文件中提取任何原子")
            
            return np.array(atom_list), atom_type_list
            
        except Exception as primary_error:
            # BioPython解析失败，尝试手动解析
            print(f"BioPython解析失败: {primary_error}")
            print("尝试使用手动解析...")
            
            if file_ext == ".pdb":
                return parse_pdb_file_manually(pdb_file, backbone_only)
            elif file_ext == ".cif":
                return parse_cif_file_manually(pdb_file, backbone_only)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
                
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        raise

# 手动解析PDB文件
def parse_pdb_file_manually(pdb_file, backbone_only=False):
    """手动解析PDB文件，提取原子坐标和类型"""
    atom_list = []
    atom_type_list = []
    
    with open(pdb_file, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                try:
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip()
                    
                    # 如果元素字段为空，从原子名称推断
                    if not element:
                        element = atom_name[0]
                    
                    # 只处理骨架原子
                    if backbone_only and atom_name not in ["CA", "C", "N"]:
                        continue
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atom_list.append([x, y, z])
                    atom_type_list.append(element)
                except Exception as e:
                    print(f"解析行时出错: {e}, 行内容: {line}")
                    continue
    
    if not atom_list:
        raise ValueError("未能从PDB文件中提取任何原子")
    
    print(f"手动从PDB文件提取了{len(atom_list)}个原子")
    return np.array(atom_list), atom_type_list

# 手动解析CIF文件
def parse_cif_file_manually(cif_file, backbone_only=False):
    """手动解析CIF文件，提取原子坐标和类型"""
    atom_list = []
    atom_type_list = []
    
    # 定义解析浮点值的辅助函数
    def parse_float_value(value_str):
        try:
            # 处理CIF中的特殊值如"?"或"."
            if value_str == "?" or value_str == ".":
                return 0.0
            return float(value_str)
        except ValueError:
            print(f"无法解析浮点值: {value_str}")
            return 0.0
    
    with open(cif_file, 'r', errors='ignore') as f:
        lines = f.readlines()
        
        # 查找atom_site表位置
        atom_site_start = -1
        for i, line in enumerate(lines):
            if line.startswith("_atom_site."):
                atom_site_start = i
                break
        
        if atom_site_start == -1:
            raise ValueError("在CIF文件中未找到atom_site表")
        
        # 解析列标题
        headers = {}
        current_index = atom_site_start
        while current_index < len(lines) and lines[current_index].startswith("_atom_site."):
            header = lines[current_index].strip()
            field_name = header.split(".")[1]
            headers[field_name] = current_index - atom_site_start
            current_index += 1
        
        # 解析原子坐标数据
        while current_index < len(lines):
            line = lines[current_index].strip()
            if not line or line.startswith("#") or line.startswith("loop_"):
                break
                
            parts = line.split()
            if len(parts) < max(headers.values()) + 1:
                current_index += 1
                continue
                
            try:
                atom_name = parts[headers.get("label_atom_id", headers.get("auth_atom_id", -1))].strip("'\"")
                element = parts[headers.get("type_symbol", -1)].strip("'\"") if "type_symbol" in headers else ""
                
                # 如果元素符号不可用，从原子名称推断
                if element == "" or element == -1:
                    element = atom_name[0]
                
                # 只处理骨架原子
                if backbone_only and atom_name not in ["CA", "C", "N"]:
                    current_index += 1
                    continue
                
                # 获取坐标
                x_idx = headers.get("Cartn_x", -1)
                y_idx = headers.get("Cartn_y", -1)
                z_idx = headers.get("Cartn_z", -1)
                
                if x_idx != -1 and y_idx != -1 and z_idx != -1:
                    x = parse_float_value(parts[x_idx])
                    y = parse_float_value(parts[y_idx])
                    z = parse_float_value(parts[z_idx])
                    
                    atom_list.append([x, y, z])
                    atom_type_list.append(element)
            except Exception as e:
                print(f"解析CIF行时出错: {e}, 行内容: {line}")
            
            current_index += 1
    
    if not atom_list:
        raise ValueError("未能从CIF文件中提取任何原子")
    
    print(f"手动从CIF文件提取了{len(atom_list)}个原子")
    return np.array(atom_list), atom_type_list

# 计算质心
def calculate_centre_of_mass(atom_list, atom_type_list):
    """计算原子的质心"""
    if len(atom_list) != len(atom_type_list):
        raise ValueError("原子列表和原子类型列表长度不匹配")
    
    total_mass = 0.0
    weighted_sum = np.zeros(3)
    
    for i, atom_coord in enumerate(atom_list):
        atom_type = atom_type_list[i]
        mass = atom_mass_dict.get(atom_type, 12.011)  # 默认碳原子质量
        
        total_mass += mass
        weighted_sum += mass * atom_coord
    
    if total_mass == 0:
        raise ValueError("总质量为零，无法计算质心")
    
    return weighted_sum / total_mass

# 计算蛋白质地图的尺寸和原点
def prot2map(atom_list, atom_type_list, voxel_size, resolution=None):
    """计算蛋白质地图的尺寸和原点"""
    # 计算质心
    x_co_m, y_co_m, z_co_m = calculate_centre_of_mass(atom_list, atom_type_list)
    
    # 计算蛋白质边界
    min_coords = np.min(atom_list, axis=0)
    max_coords = np.max(atom_list, axis=0)
    
    # 计算边界尺寸
    x_size = max_coords[0] - min_coords[0]
    y_size = max_coords[1] - min_coords[1]
    z_size = max_coords[2] - min_coords[2]
    
    # 添加解析度相关的填充
    if resolution is not None:
        padding = resolution * 3  # 添加3倍分辨率的填充
    else:
        padding = 10.0  # 默认填充10埃
    
    # 添加填充
    min_coords -= padding
    max_coords += padding
    
    # 更新尺寸
    x_size = max_coords[0] - min_coords[0]
    y_size = max_coords[1] - min_coords[1]
    z_size = max_coords[2] - min_coords[2]
    
    # 计算网格尺寸
    nx = int(np.ceil(x_size / voxel_size[0]))
    ny = int(np.ceil(y_size / voxel_size[1]))
    nz = int(np.ceil(z_size / voxel_size[2]))
    
    # 计算原点
    origin = min_coords
    
    print(f"计算的地图尺寸: {nx} x {ny} x {nz}")
    print(f"计算的原点: {origin}")
    
    return (nx, ny, nz), origin

# 原子坐标映射到网格位置
def mapGridPosition(origin, voxel_size, box_size, atom_coord):
    """将原子坐标映射到网格位置"""
    # 计算相对坐标
    rel_coord = atom_coord - origin
    
    # 计算网格位置
    i = int(rel_coord[0] / voxel_size[0])
    j = int(rel_coord[1] / voxel_size[1])
    k = int(rel_coord[2] / voxel_size[2])
    
    # 检查是否在网格内
    if 0 <= i < box_size[0] and 0 <= j < box_size[1] and 0 <= k < box_size[2]:
        return (i, j, k)
    else:
        return None

# 创建原子叠加地图
def make_atom_overlay_map(origin, voxel_size, box_size, atom_list, atom_type_list, atom_mass_dict):
    """创建原子叠加地图"""
    # 初始化密度图
    density_map = np.zeros(box_size, dtype=np.float32)
    
    # 处理每个原子
    for i, atom_coord in enumerate(atom_list):
        # 获取网格位置
        grid_pos = mapGridPosition(origin, voxel_size, box_size, atom_coord)
        
        if grid_pos is not None:
            # 获取原子质量
            atom_type = atom_type_list[i]
            mass = atom_mass_dict.get(atom_type, 12.011)  # 默认碳原子质量
            
            # 在密度图中添加原子密度
            density_map[grid_pos[0], grid_pos[1], grid_pos[2]] += mass
    
    return density_map

# 写入MRC文件
def write_mrc_file(data, origin, voxel_size, mrc_file):
    """将数据写入MRC文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(mrc_file), exist_ok=True)
    
    # 创建MRC文件
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        # 设置数据
        mrc.set_data(data.astype(np.float32))
        
        # 设置体素大小
        vsize = mrc.voxel_size
        vsize.flags.writeable = True
        vsize.x = float(voxel_size[0])
        vsize.y = float(voxel_size[1])
        vsize.z = float(voxel_size[2])
        mrc.voxel_size = vsize
        
        # 设置原点
        mrc.header.origin.x = float(origin[0])
        mrc.header.origin.y = float(origin[1])
        mrc.header.origin.z = float(origin[2])
        
        # 设置标准轴顺序
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3
        
        # 更新头信息统计
        mrc.update_header_stats()
    
    print(f"成功将数据写入MRC文件: {mrc_file}")

# 模糊地图(傅里叶空间)
def blur_map(data, resolution, sigma_coeff):
    """使用傅里叶空间高斯滤波模糊地图"""
    # 计算sigma
    sigma = resolution * sigma_coeff
    
    # 执行傅里叶变换
    ft = fftn(data)
    
    # 应用高斯滤波
    filtered_ft = fourier_gaussian(ft, sigma=sigma)
    
    # 执行逆傅里叶变换
    blurred_data = np.real(ifftn(filtered_ft))
    
    return blurred_data

# 模糊地图(实空间)
def blur_map_real_space(data, resolution, sigma_coeff):
    """使用实空间高斯滤波模糊地图"""
    # 计算sigma
    sigma = resolution * sigma_coeff
    
    # 应用高斯滤波
    blurred_data = gaussian_filter(data, sigma=sigma)
    
    return blurred_data

# 归一化地图
def normalize_map(map_data):
    """归一化地图"""
    mean_val = np.mean(map_data)
    std_val = np.std(map_data)
    
    if std_val == 0:
        print("警告: 标准差为零，无法归一化")
        return map_data
    
    # 归一化
    normalized_data = (map_data - mean_val) / std_val
    
    return normalized_data

# 重采样地图
def resample_by_box_size(data, box_size):
    """重采样地图以匹配指定的盒子大小"""
    # 计算缩放系数
    zoom_factors = (box_size[0] / data.shape[0], 
                    box_size[1] / data.shape[1], 
                    box_size[2] / data.shape[2])
    
    # 执行重采样
    resampled_data = zoom(data, zoom_factors, order=1)
    
    return resampled_data

# PDB转体积地图
def pdb2vol(
        input_pdb,
        resolution,
        output_mrc=None,
        ref_map=None,
        sigma_coeff=0.356,
        real_space=False,
        normalize=True,
        backbone_only=False,
        contour=False,
        apply_contour=True,
        bin_mask=False,
        return_data=False,
):
    """
    将PDB或CIF文件转换为体积密度图
    
    参数:
    - input_pdb: 输入PDB或CIF文件路径
    - resolution: 输出地图的分辨率
    - output_mrc: 输出MRC文件路径
    - ref_map: 参考密度图路径
    - sigma_coeff: 高斯模糊的sigma系数
    - real_space: 是否在实空间执行高斯模糊
    - normalize: 是否归一化输出地图
    - backbone_only: 是否只使用骨架原子
    - contour: 阈值水平
    - apply_contour: 是否应用阈值
    - bin_mask: 是否二值化输出地图
    - return_data: 是否返回地图数据
    
    返回:
    - 如果return_data为True，返回地图数据；否则返回None
    """
    # 检查文件扩展名
    file_ext = os.path.splitext(input_pdb)[1].lower()
    if file_ext not in [".pdb", ".cif"]:
        raise ValueError(f"输入文件必须是PDB或CIF格式。当前文件扩展名: {file_ext}")
    
    print(f"处理输入文件: {input_pdb}，文件类型: {'CIF' if file_ext == '.cif' else 'PDB'}")
    
    try:
        # 获取原子列表和类型
        atoms, types = get_atom_list(input_pdb, backbone_only=backbone_only)
    except Exception as e:
        raise ValueError(f"无法解析{file_ext}文件: {e}")

    if len(atoms) == 0:
        raise ValueError("在输入文件中未找到原子")
    if len(atoms) != len(types):
        raise ValueError("原子数量和原子类型数量不匹配")

    # 确定体素大小和尺寸
    if not ref_map:
        r = np.clip(resolution / 4.0, a_min=1.0, a_max=3.5)
        voxel_size = np.array([r, r, r])
        dims, origin = prot2map(atoms, types, voxel_size, resolution)
    else:
        print(f"使用参考密度图: {ref_map}")
        try:
            with mrcfile.open(ref_map, permissive=True) as mrc:
                voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
                dims = mrc.data.shape
                origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
                print(f"参考密度图尺寸: {dims}")
                print(f"参考密度图体素大小: {voxel_size}")
                print(f"参考密度图原点: {origin}")
        except Exception as e:
            print(f"读取参考密度图失败: {e}")
            print("将使用自动计算的尺寸和原点")
            r = np.clip(resolution / 4.0, a_min=1.0, a_max=3.5)
            voxel_size = np.array([r, r, r])
            dims, origin = prot2map(atoms, types, voxel_size, resolution)

    # 计算新的盒子大小
    x_s = int(dims[2] * voxel_size[2])
    y_s = int(dims[1] * voxel_size[1])
    z_s = int(dims[0] * voxel_size[0])

    # 计算新的体素大小
    new_voxel_size = np.array(
        [voxel_size[2] * dims[2] / x_s, voxel_size[1] * dims[1] / y_s, voxel_size[0] * dims[0] / z_s])

    # 创建原子叠加地图
    map_data = make_atom_overlay_map(origin, new_voxel_size, (z_s, y_s, x_s), atoms, types, atom_mass_dict)

    # 执行高斯模糊
    if resolution * sigma_coeff / new_voxel_size[0] >= 1:
        if real_space:
            blurred_data = blur_map_real_space(map_data, resolution, sigma_coeff)
        else:
            blurred_data = blur_map(map_data, resolution, sigma_coeff)
    else:
        print("警告: 不执行模糊处理，因为分辨率相对于体素大小太高")
        blurred_data = map_data  # 不模糊

    # 重采样以匹配原始尺寸
    blurred_data = resample_by_box_size(blurred_data, dims)

    # 归一化
    if normalize:
        blurred_data = normalize_map(blurred_data)
        # 输出归一化后的统计信息
        min_val = np.min(blurred_data)
        max_val = np.max(blurred_data)
        mean_val = np.mean(blurred_data)
        std_val = np.std(blurred_data)
        print(f"归一化后的统计信息: 最小值={min_val:.6f}, 最大值={max_val:.6f}, 均值={mean_val:.6f}, 标准差={std_val:.6f}")
        
        # 计算正负值比例
        positive_pct = np.sum(blurred_data > 0) / blurred_data.size * 100
        negative_pct = np.sum(blurred_data < 0) / blurred_data.size * 100
        print(f"正值占比: {positive_pct:.2f}%, 负值占比: {negative_pct:.2f}%")

    # 记录contour和bin_mask处理前的状态
    print(f"在contour和bin_mask处理前，数据包含负值: {np.any(blurred_data < 0)}")
    print(f"contour参数值: {contour}, apply_contour: {apply_contour}")
    print(f"bin_mask参数值: {bin_mask}")

    # 应用阈值
    if contour and apply_contour:
        print(f"执行contour处理，contour阈值: {contour}")
        blurred_data = np.where(blurred_data > contour, blurred_data, 0)
        print(f"contour处理后，数据是否包含负值: {np.any(blurred_data < 0)}")

    # 二值化
    if bin_mask:
        print("执行bin_mask操作")
        # 二值化以获取掩码
        blurred_data = np.where(blurred_data > 0, 1, 0)
        print(f"bin_mask处理后，数据是否包含负值: {np.any(blurred_data < 0)}")

    # 写入MRC文件
    if output_mrc is not None:
        print(f"写入MRC文件前，数据是否包含负值: {np.any(blurred_data < 0)}")
        print(f"写入MRC文件前，最小值: {np.min(blurred_data):.6f}")
        write_mrc_file(blurred_data, origin, voxel_size, output_mrc)

    # 返回数据（如果需要）
    if return_data:
        return blurred_data

# 计算质心
def calculate_centre_of_mass(atom_list, atom_type_list):
    """计算原子的质心"""
    if len(atom_list) != len(atom_type_list):
        raise ValueError("原子列表和原子类型列表长度不匹配")
    
    total_mass = 0.0
    weighted_sum = np.zeros(3)
    
    for i, atom_coord in enumerate(atom_list):
        atom_type = atom_type_list[i]
        mass = atom_mass_dict.get(atom_type, 12.011)  # 默认碳原子质量
        
        total_mass += mass
        weighted_sum += mass * atom_coord
    
    if total_mass == 0:
        raise ValueError("总质量为零，无法计算质心")
    
    return weighted_sum / total_mass

# 计算蛋白质地图的尺寸和原点
def prot2map(atom_list, atom_type_list, voxel_size, resolution=None):
    """计算蛋白质地图的尺寸和原点"""
    # 计算质心
    x_co_m, y_co_m, z_co_m = calculate_centre_of_mass(atom_list, atom_type_list)
    
    # 计算蛋白质边界
    min_coords = np.min(atom_list, axis=0)
    max_coords = np.max(atom_list, axis=0)
    
    # 计算边界尺寸
    x_size = max_coords[0] - min_coords[0]
    y_size = max_coords[1] - min_coords[1]
    z_size = max_coords[2] - min_coords[2]
    
    # 添加解析度相关的填充
    if resolution is not None:
        padding = resolution * 3  # 添加3倍分辨率的填充
    else:
        padding = 10.0  # 默认填充10埃
    
    # 添加填充
    min_coords -= padding
    max_coords += padding
    
    # 更新尺寸
    x_size = max_coords[0] - min_coords[0]
    y_size = max_coords[1] - min_coords[1]
    z_size = max_coords[2] - min_coords[2]
    
    # 计算网格尺寸
    nx = int(np.ceil(x_size / voxel_size[0]))
    ny = int(np.ceil(y_size / voxel_size[1]))
    nz = int(np.ceil(z_size / voxel_size[2]))
    
    # 计算原点
    origin = min_coords
    
    print(f"计算的地图尺寸: {nx} x {ny} x {nz}")
    print(f"计算的原点: {origin}")
    
    return (nx, ny, nz), origin

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量将PDB文件转换为MRC，再转换为CryFold可用的CIF格式")
    parser.add_argument("--input_dir", required=True, help="输入目录，包含多个蛋白质子目录")
    parser.add_argument("--protein_list", required=True, help="蛋白质名称列表文件，每行一个")
    parser.add_argument("--atom_type", choices=["ca_only", "backbone"], default="ca_only", 
                        help="要提取的原子类型：仅CA原子或所有骨架原子(CA,C,N)")
    parser.add_argument("--resolution", type=float, default=3.0, help="生成密度图的分辨率")
    parser.add_argument("--threshold", type=float, default=0.5, help="密度阈值")
    parser.add_argument("--min_distance", type=float, default=3.8, help="最小原子间距（埃）")
    parser.add_argument("--use_fps", action="store_true", help="是否使用最远点取样")
    parser.add_argument("--n_samples", type=int, default=1000, help="最远点取样的采样点数量")
    return parser.parse_args()

def filter_backbone(input_path, backbone_output_path, only_ca=True):
    """
    提取PDB或CIF文件中的骨架原子
    
    Args:
        input_path (str): 输入PDB或CIF文件路径
        backbone_output_path (str): 输出只包含骨架原子的文件路径
        only_ca (bool): 是否只提取CA原子
    """
    # 定义骨架原子列表
    if only_ca:
        backbone_list = ["CA"]
    else:
        backbone_list = ["CA", "C", "N"]
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # 根据文件扩展名选择适当的解析器
    if file_ext == '.cif':
        parser = MMCIFParser(QUIET=True)
        print(f"使用MMCIFParser解析CIF文件: {input_path}")
    else:
        parser = PDBParser(QUIET=True)
        print(f"使用PDBParser解析PDB文件: {input_path}")
        
    try:
        from Bio.PDB import PDBIO
        io = PDBIO()
        
        structure = parser.get_structure("protein", input_path)
        # 创建只包含骨架原子的新结构
        for model in structure:
            for chain in model:
                atom_indices_to_detach = []
                for i, atom in enumerate(chain.get_atoms()):
                    atom_name = atom.get_name()
                    if atom_name not in backbone_list:
                        # 收集不是骨架的原子的索引
                        residue = atom.get_parent()
                        atom_indices_to_detach.append((residue, atom.get_name()))
        
        # 移除非骨架原子
        for residue, atom_name in atom_indices_to_detach:
            if atom_name in residue:
                residue.detach_child(atom_name)
        
        # 将过滤后的结构写入PDB文件
        io.set_structure(structure)
        io.save(backbone_output_path)
        
        atom_count = sum(1 for model in structure for chain in model 
                         for residue in chain for atom in residue)
        print(f"已从文件提取{len(backbone_list)}种原子类型，保存了{atom_count}个原子")
        
        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

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
    - ca_coords: 均值漂移聚类后的坐标点
    - ca_coords_before_pruning: 处理前的点云
    """
    def points_to_grid(points, shape):
        """将点集转换为密度图"""
        grid = np.zeros(shape, dtype=np.float32)
        x, y, z = points[:, 0].astype(np.int32), points[:, 1].astype(np.int32), points[:, 2].astype(np.int32)
        
        # 过滤掉超出范围的点
        valid_mask = (x >= 0) & (x < shape[0]) & (y >= 0) & (y < shape[1]) & (z >= 0) & (z < shape[2])
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        grid[x, y, z] = 1.0
        return grid
    
    # 对密度图应用阈值
    mask = grid > threshold
    
    # 如果没有超过阈值的体素，则返回空列表
    if not np.any(mask):
        print(f"警告: 使用阈值 {threshold} 未找到任何非零体素")
        return [], []
    
    # 创建超过阈值的体素列表
    points_idx = np.argwhere(mask)
    
    if points_idx.shape[0] == 0:
        return [], []
    
    # 创建体素坐标网格
    mesh = get_lattice_meshgrid_np(grid.shape)
    
    # 提取超过阈值的体素坐标
    points = mesh[mask]
    print(f"超过阈值 {threshold} 的体素数量: {len(points)}")
    
    # 创建初始点云，保存用于调试
    points_before_pruning = points.copy()
    
    # 如果点太少，直接返回
    if len(points) <= 10:
        return points, points_before_pruning
    
    # 使用KDTree执行均值漂移聚类
    tree = cKDTree(points)
    
    # 迭代直到收敛
    max_iterations = 20
    for i in range(max_iterations):
        # 找出每个点邻域内的所有点
        new_points = []
        for p in points:
            # 查询半径内的所有邻居
            indices = tree.query_ball_point(p, neighbour_distance_threshold)
            if len(indices) > 0:
                # 计算邻居的均值作为新点位置
                new_point = np.mean(points[indices], axis=0)
                new_points.append(new_point)
        
        new_points = np.array(new_points)
        
        # 计算移动距离
        if len(new_points) == len(points):
            distances = np.linalg.norm(new_points - points, axis=1)
            mean_distance = np.mean(distances)
            print(f"迭代 {i+1}/{max_iterations}, 平均移动距离: {mean_distance:.6f}")
            
            # 检查是否收敛
            if mean_distance < 0.01:
                print(f"均值漂移已收敛，总迭代次数: {i+1}")
                break
        
        points = new_points
        tree = cKDTree(points)
    
    # 通过密度聚类移除噪声点
    pruned_points = []
    min_neighbors = 3  # 至少需要3个邻居才被视为有效点
    
    for p in points:
        # 查询附近的点数量
        count = len(tree.query_ball_point(p, neighbour_distance_threshold))
        if count >= min_neighbors:
            pruned_points.append(p)
    
    pruned_points = np.array(pruned_points)
    print(f"聚类后的点数: {len(pruned_points)}")
    
    # 对点云进行距离过滤，移除太近的点
    if len(pruned_points) > 0:
        # 计算每对点之间的距离
        dists = squareform(pdist(pruned_points))
        
        # 将自身与自身的距离设为无穷大，避免被选择
        np.fill_diagonal(dists, np.inf)
        
        keep_indices = []
        remaining_indices = set(range(len(pruned_points)))
        
        while remaining_indices:
            # 随机选择一个起始点
            current_idx = remaining_indices.pop()
            keep_indices.append(current_idx)
            
            # 找出所有与当前点距离小于阈值的点
            too_close = np.where(dists[current_idx] < neighbour_distance_threshold)[0]
            
            # 从剩余索引中移除这些点
            remaining_indices -= set(too_close)
        
        # 最终的过滤点集
        ca_coords = pruned_points[keep_indices]
        
        # 如果需要，则应用最远点取样
        if use_fps and len(ca_coords) > n_samples:
            ca_coords = farthest_point_sampling(ca_coords, n_samples)
        
        print(f"最终CA原子数: {len(ca_coords)}")
        
        # 如果提供了输出目录，保存处理前后的点云密度图
        if output_dir and voxel_size is not None and global_origin is not None:
            # 保存处理前的点云
            before_grid = points_to_grid(points_before_pruning, grid.shape)
            before_map_path = os.path.join(output_dir, "points_before_pruning.mrc")
            save_dens_map(before_map_path, before_grid, voxel_size, global_origin)
            
            # 保存聚类后的点云
            after_grid = points_to_grid(ca_coords, grid.shape)
            after_map_path = os.path.join(output_dir, "points_after_pruning.mrc")
            save_dens_map(after_map_path, after_grid, voxel_size, global_origin)
            
            print(f"已保存处理前后的点云密度图到: {output_dir}")
        
        return ca_coords, points_before_pruning
    else:
        return [], points_before_pruning

def points_to_cif(path_to_save, points):
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

def pdb_to_mrc(input_pdb, output_mrc, resolution=3.0, only_ca=True):
    """
    将PDB或CIF文件转换为MRC密度图
    
    参数：
    - input_pdb: 输入PDB或CIF文件路径
    - output_mrc: 输出MRC文件路径
    - resolution: 分辨率
    - only_ca: 是否只提取CA原子
    
    返回：
    - 是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_mrc), exist_ok=True)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建临时文件存储过滤后的骨架原子
            basename = os.path.splitext(os.path.basename(input_pdb))[0]
            backbone_pdb = os.path.join(temp_dir, f"{basename}_backbone.pdb")
            
            # 提取骨架原子
            if not filter_backbone(input_pdb, backbone_pdb, only_ca):
                print(f"错误: 无法从文件 {input_pdb} 提取骨架原子")
                return False
            
            # 使用内部的pdb2vol函数将PDB转换为MRC
            try:
                pdb2vol(
                    input_pdb=backbone_pdb,
                    output_mrc=output_mrc,
                    resolution=resolution,
                    sigma_coeff=0.356,  # 使标准高斯宽度等于分辨率
                    normalize=True,
                    backbone_only=True,  # 已经过滤了骨架原子，这里设为True只是为了确保
                    real_space=False,
                    contour=False,
                    apply_contour=False,
                    bin_mask=False,
                    return_data=False
                )
                print(f"成功将PDB转换为MRC: {output_mrc}")
                return True
            except Exception as e:
                print(f"使用pdb2vol转换时出错: {e}")
                return False
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"转换PDB到MRC时出错: {e}")
        return False

def mrc_to_cif(input_mrc, output_cif, threshold=0.5, min_distance=3.8, use_fps=False, n_samples=1000):
    """
    将MRC密度图转换为CIF文件
    
    参数：
    - input_mrc: 输入MRC文件路径
    - output_cif: 输出CIF文件路径
    - threshold: 密度阈值
    - min_distance: 最小原子间距（埃）
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    
    返回：
    - 是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_cif), exist_ok=True)
        
        # 加载MRC文件
        density, voxel_size, origin = load_mrc(input_mrc)
        
        # 将密度图转换为点云
        ca_coords, _ = grid_to_points(
            density, 
            threshold, 
            min_distance, 
            os.path.dirname(output_cif), 
            voxel_size, 
            origin,
            use_fps,
            n_samples
        )
        
        if not ca_coords or len(ca_coords) == 0:
            print(f"错误: 使用阈值 {threshold} 未能提取到任何点")
            
            # 尝试降低阈值
            reduced_threshold = threshold / 2.0
            print(f"尝试降低阈值到 {reduced_threshold}")
            
            ca_coords, _ = grid_to_points(
                density, 
                reduced_threshold, 
                min_distance, 
                os.path.dirname(output_cif), 
                voxel_size, 
                origin,
                use_fps,
                n_samples
            )
            
            if not ca_coords or len(ca_coords) == 0:
                print(f"错误: 即使降低阈值到 {reduced_threshold} 仍未能提取到任何点")
                return False
        
        # 将体素坐标转换为真实坐标
        real_coords = ca_coords * voxel_size[None] + origin[None]
        
        # 将点云转换为CIF文件
        points_to_cif(output_cif, real_coords)
        
        print(f"成功将MRC转换为CIF: {output_cif}")
        return True
    
    except Exception as e:
        print(f"转换MRC到CIF时出错: {e}")
        return False

def pdb_to_mrc_to_cif(input_pdb, output_cif, resolution=3.0, threshold=0.5, min_distance=3.8, only_ca=True, use_fps=False, n_samples=1000):
    """
    将PDB文件转换为MRC，再将MRC转换为CIF文件
    
    参数：
    - input_pdb: 输入PDB或CIF文件路径
    - output_cif: 输出CIF文件路径
    - resolution: 分辨率
    - threshold: 密度阈值
    - min_distance: 最小原子间距（埃）
    - only_ca: 是否只提取CA原子
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    
    返回：
    - 是否成功
    """
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建临时MRC文件
            basename = os.path.splitext(os.path.basename(input_pdb))[0]
            temp_mrc = os.path.join(temp_dir, f"{basename}_temp.mrc")
            
            # 步骤1: PDB到MRC
            if not pdb_to_mrc(input_pdb, temp_mrc, resolution, only_ca):
                print(f"错误: 无法将PDB转换为MRC: {input_pdb}")
                return False
            
            # 步骤2: MRC到CIF
            if not mrc_to_cif(temp_mrc, output_cif, threshold, min_distance, use_fps, n_samples):
                print(f"错误: 无法将MRC转换为CIF: {temp_mrc}")
                return False
            
            print(f"成功将PDB转换为CIF: {output_cif}")
            return True
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"转换PDB到CIF时出错: {e}")
        return False

def batch_process(input_dir, protein_list_file, atom_type="ca_only", resolution=3.0, threshold=0.5, min_distance=3.8, use_fps=False, n_samples=1000):
    """
    批量处理PDB文件，转换为CIF格式
    
    参数：
    - input_dir: 输入目录，包含多个蛋白质子目录
    - protein_list_file: 蛋白质名称列表文件，每行一个
    - atom_type: 要提取的原子类型，ca_only或backbone
    - resolution: 生成密度图的分辨率
    - threshold: 密度阈值
    - min_distance: 最小原子间距（埃）
    - use_fps: 是否使用最远点取样
    - n_samples: 最远点取样的采样点数量
    """
    # 加载蛋白质名称列表
    with open(protein_list_file, 'r') as f:
        protein_names = [line.strip() for line in f if line.strip()]
    
    print(f"将处理以下 {len(protein_names)} 个蛋白质:")
    for name in protein_names:
        print(f"  - {name}")
    
    # 处理每个蛋白质文件夹
    processed_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            # 检查文件夹名格式是否符合 PDB-XXXX-EMD-XXXXX
            if dir_name.startswith("PDB-") and "-EMD-" in dir_name:
                # 提取PDB代码
                pdb_code = dir_name.split("-")[1].lower()
                
                # 检查是否在处理列表中
                if pdb_code in protein_names:
                    print(f"\n处理蛋白质: {pdb_code} ({dir_name})")
                    
                    # 查找该文件夹中的PDB文件
                    pdb_dir = os.path.join(root, dir_name)
                    pdb_file = None
                    
                    # 按照优先级查找PDB文件
                    potential_pdb_files = [
                        os.path.join(pdb_dir, f"{pdb_code}.pdb"),
                        os.path.join(pdb_dir, f"PDB-{pdb_code}.pdb"),
                        os.path.join(pdb_dir, dir_name + ".pdb")
                    ]
                    
                    for potential_file in potential_pdb_files:
                        if os.path.exists(potential_file):
                            pdb_file = potential_file
                            break
                    
                    # 如果没有找到标准命名的PDB文件，则搜索文件夹中的所有PDB文件
                    if pdb_file is None:
                        for potential_file in os.listdir(pdb_dir):
                            if potential_file.lower().endswith('.pdb'):
                                pdb_file = os.path.join(pdb_dir, potential_file)
                                break
                    
                    if pdb_file is not None:
                        # 定义输出CIF文件路径
                        output_cif = os.path.join(pdb_dir, f"{pdb_code}_point.cif")
                        
                        print(f"  找到PDB文件: {pdb_file}")
                        print(f"  将转换为CIF文件: {output_cif}")
                        
                        try:
                            # 转换PDB为MRC再到CIF
                            result = pdb_to_mrc_to_cif(
                                pdb_file, 
                                output_cif, 
                                resolution, 
                                threshold, 
                                min_distance, 
                                atom_type == "ca_only",
                                use_fps,
                                n_samples
                            )
                            if result:
                                processed_count += 1
                                print(f"  转换成功: {pdb_code}")
                            else:
                                error_count += 1
                                print(f"  转换失败: {pdb_code}")
                        except Exception as e:
                            error_count += 1
                            print(f"  处理文件时出错: {e}")
                    else:
                        error_count += 1
                        print(f"  错误: 在文件夹 {pdb_dir} 中找不到PDB文件")
    
    print(f"\n批处理完成！成功转换: {processed_count}, 失败: {error_count}")

def main():
    args = parse_args()
    batch_process(
        args.input_dir, 
        args.protein_list, 
        args.atom_type, 
        args.resolution, 
        args.threshold, 
        args.min_distance, 
        args.use_fps, 
        args.n_samples
    )

if __name__ == "__main__":
    main()