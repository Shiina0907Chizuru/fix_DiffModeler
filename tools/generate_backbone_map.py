#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成骨架密度图工具

这个脚本用于从PDB文件生成蛋白质骨架的密度图。
它首先提取PDB文件中的骨架原子（CA、C、N），然后将其转换为体积密度图。
"""

import os
import argparse
import sys
import numpy as np
import re
from pathlib import Path
import glob

# 添加项目根目录到sys.path，确保能导入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接定义filter_backbone函数，避免从modeling.pdb_utils导入
def filter_backbone(input_pdb_path, backbone_pdb_path):
    backbone_list = ["CA", "C", "N"]
    # support DNA/RNA template fitting
    backbone_list_drna = ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
    backbone_list += backbone_list_drna
    with open(input_pdb_path, 'r') as file:
        with open(backbone_pdb_path, 'w') as wfile:
            for line in file:
                if line.startswith("ATOM"):
                    chain_name = line[21]
                    atom_name = line[12:16].replace(" ", "")
                    if atom_name in backbone_list:
                        wfile.write(line)

from ops.pdb2vol import pdb2vol
import mrcfile


def parse_protein_info_file(info_file_path):
    """
    解析包含蛋白质EMD ID、Contour Level和Resolution的文本文件
    
    Args:
        info_file_path (str): 包含蛋白质信息的文本文件路径
    
    Returns:
        dict: 以EMD ID为键，包含contour_level和resolution的字典
    """
    protein_info = {}
    
    try:
        with open(info_file_path, 'r') as f:
            for line in f:
                # 使用正则表达式提取EMD ID、Contour Level和Resolution
                match = re.match(r'([^\s:]+):\s+EMD-(\d+),\s+Contour\s+Level:\s+([\d.]+),\s+Resolution:\s+([\d.]+)\s+A', line)
                if match:
                    protein_code = match.group(1)  # 蛋白质代码，如3j3r
                    emd_id = f"EMD-{match.group(2)}"  # EMD ID，如EMD-5610
                    contour_level = float(match.group(3))  # Contour Level，如1.5
                    resolution = float(match.group(4))  # Resolution，如9.4
                    
                    protein_info[emd_id] = {
                        'contour_level': contour_level,
                        'resolution': resolution,
                        'protein_code': protein_code
                    }
                    print(f"解析到蛋白质信息: {protein_code}, {emd_id}, Contour Level: {contour_level}, Resolution: {resolution}")
        
        if not protein_info:
            print("警告: 未从文件中解析到任何蛋白质信息")
    except Exception as e:
        print(f"解析蛋白质信息文件时出错: {e}")
    
    return protein_info


def get_protein_info_by_emd_id(protein_info, emd_id):
    """
    根据EMD ID获取蛋白质信息
    
    Args:
        protein_info (dict): 包含蛋白质信息的字典
        emd_id (str): EMD ID
    
    Returns:
        dict: 包含contour_level和resolution的字典，未找到则返回None
    """
    # 确保EMD ID格式一致
    if not emd_id.startswith("EMD-"):
        emd_id = f"EMD-{emd_id}"
    
    return protein_info.get(emd_id)


def generate_backbone_density(input_pdb, output_mrc, resolution=1.0, backbone_only=True, normalize=True, contour_level=None, reference_map=None):
    """
    从PDB文件生成骨架密度图

    Args:
        input_pdb (str): 输入PDB文件路径
        output_mrc (str): 输出密度图MRC文件路径
        resolution (float): 密度图分辨率
        backbone_only (bool): 是否只包含骨架原子
        normalize (bool): 是否对密度图进行归一化
        contour_level (float, optional): 密度图的等值面水平
        reference_map (str, optional): 参考密度图路径，用于保持尺寸一致
    
    Returns:
        str: 输出密度图路径
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_mrc)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建临时PDB文件存储骨架原子
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    backbone_pdb = os.path.join(temp_dir, f"{Path(input_pdb).stem}_backbone.pdb")
    
    # 提取骨架原子
    filter_backbone(input_pdb, backbone_pdb)
    print(f"已提取骨架原子到: {backbone_pdb}")
    
    # 转换为体积密度图
    pdb2vol(
        input_pdb=backbone_pdb,
        resolution=resolution,
        output_mrc=output_mrc,
        normalize=normalize,
        backbone_only=backbone_only,
        contour=contour_level,
        ref_map=reference_map  # 使用参考密度图保持尺寸一致
    )
    print(f"已生成骨架密度图: {output_mrc}")
    
    # 显示密度图信息
    try:
        with mrcfile.open(output_mrc, permissive=True) as mrc:
            data = mrc.data
            print(f"密度图大小: {data.shape}")
            print(f"密度值范围: {np.min(data)} to {np.max(data)}")
    except Exception as e:
        print(f"读取密度图信息时出错: {e}")
    
    return output_mrc


def process_data_directory(data_root, protein_info_file, segment_pattern="{pdb_id}_segment.mrc"):
    """
    处理数据目录，为每个蛋白质生成骨架密度图
    
    Args:
        data_root (str): 数据根目录，包含所有蛋白质子目录
        protein_info_file (str): 蛋白质信息文件路径
        segment_pattern (str, optional): 参考密度图文件名模式，用于查找segment map
    """
    # 解析蛋白质信息文件
    protein_info = parse_protein_info_file(protein_info_file)
    
    # 获取数据根目录下的所有PDB子目录
    pdb_dirs = glob.glob(os.path.join(data_root, "PDB-*-EMD-*"))
    
    print(f"找到{len(pdb_dirs)}个蛋白质目录")
    
    # 处理每个蛋白质目录
    success_count = 0
    error_count = 0
    
    for pdb_dir in pdb_dirs:
        try:
            # 从目录名中提取蛋白质ID和EMD ID
            dir_name = os.path.basename(pdb_dir)
            match = re.match(r"PDB-([^-]+)-EMD-(\d+)", dir_name)
            if not match:
                print(f"警告: 目录名格式不正确: {dir_name}，跳过")
                continue
                
            pdb_id = match.group(1).lower()
            emd_id = match.group(2)
            
            # 查找pdb文件 - 支持多种命名模式
            possible_pdb_patterns = [
                os.path.join(pdb_dir, f"{pdb_id}*.pdb"),      # 基本名称: 3j3r.pdb
                os.path.join(pdb_dir, f"*{pdb_id}*.pdb"),     # 任何位置包含ID: PDB-3j3r.pdb
                os.path.join(pdb_dir, f"PDB-{pdb_id}.pdb"),   # 显式前缀: PDB-3j3r.pdb
                os.path.join(pdb_dir, "*.pdb")                # 任何PDB文件
            ]
            
            pdb_file = None
            for pattern in possible_pdb_patterns:
                pdb_files = glob.glob(pattern)
                if pdb_files:
                    pdb_file = pdb_files[0]
                    print(f"找到PDB文件: {pdb_file}")
                    break
                
            if not pdb_file:
                print(f"警告: 找不到{pdb_id}的PDB文件，跳过")
                continue
            
            # 获取蛋白质信息
            info = get_protein_info_by_emd_id(protein_info, f"EMD-{emd_id}")
            if not info:
                print(f"警告: 找不到EMD-{emd_id}的轮廓级别和分辨率信息，使用默认值")
                resolution = 3.0
                contour_level = None
            else:
                resolution = info["resolution"]
                contour_level = info["contour_level"]
            
            # 创建处理后的输出目录
            processed_dir = os.path.join(pdb_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # 设置输出文件路径
            output_path = os.path.join(processed_dir, f"{pdb_id}_backbone.mrc")
            
            # 自动查找参考密度图(segment map)
            # 按优先级顺序尝试多个可能的位置
            possible_segment_maps = [
                # 1. processed目录中的指定模式文件
                os.path.join(processed_dir, segment_pattern.format(pdb_id=pdb_id)),
                # 2. 主目录中的指定模式文件
                os.path.join(pdb_dir, segment_pattern.format(pdb_id=pdb_id)),
                # 3. processed目录中的任何非backbone的mrc文件
                *[f for f in glob.glob(os.path.join(processed_dir, f"{pdb_id}_*.mrc")) 
                  if not os.path.basename(f).endswith("_backbone.mrc")],
                # 4. 主目录中的任何非backbone的mrc文件
                *[f for f in glob.glob(os.path.join(pdb_dir, f"{pdb_id}_*.mrc")) 
                  if not os.path.basename(f).endswith("_backbone.mrc")]
            ]
            
            reference_map = None
            for path in possible_segment_maps:
                matching_maps = glob.glob(path) if '*' in path else [path] if os.path.exists(path) else []
                if matching_maps:
                    reference_map = matching_maps[0]
                    print(f"找到参考密度图: {reference_map}")
                    break
            
            if not reference_map:
                print(f"警告: 未找到{pdb_id}的参考密度图，将使用默认参数生成")
            
            # 生成骨架密度图
            print(f"处理 {pdb_id} (EMD-{emd_id})...")
            generate_backbone_density(
                input_pdb=pdb_file,
                output_mrc=output_path,
                resolution=resolution,
                backbone_only=True,
                normalize=True,
                contour_level=contour_level,
                reference_map=reference_map
            )
            print(f"已完成 {pdb_id} 的处理")
            success_count += 1
            
        except Exception as e:
            print(f"处理目录时出错: {pdb_dir}")
            print(f"错误详情: {str(e)}")
            error_count += 1
    
    print("所有蛋白质处理完成")
    print(f"成功处理: {success_count} 个蛋白质")
    print(f"处理失败: {error_count} 个蛋白质")


def main():
    parser = argparse.ArgumentParser(description="从PDB文件生成蛋白质骨架密度图")
    parser.add_argument("--pdb", help="输入PDB文件路径")
    parser.add_argument("--output", help="输出MRC文件路径")
    parser.add_argument("--resolution", type=float, default=3.0, help="密度图分辨率（默认: 3.0）")
    parser.add_argument("--contour", type=float, help="密度图等值面水平")
    parser.add_argument("--backbone-only", action="store_true", default=True, help="只包含骨架原子（默认: True）")
    parser.add_argument("--normalize", action="store_true", default=True, help="对密度图进行归一化（默认: True）")
    parser.add_argument("-p", "--protein-info", default=None, help="包含蛋白质信息的文本文件路径")
    parser.add_argument("-b", "--batch", action="store_true", help="批处理模式，处理指定目录下的所有蛋白质")
    parser.add_argument("-d", "--data-root", help="数据根目录，包含所有蛋白质子目录")
    parser.add_argument("-r", "--reference-map", default=None, help="参考密度图路径，用于保持尺寸一致")
    parser.add_argument("-s", "--segment-pattern", default="{pdb_id}_segment.mrc", help="参考密度图文件名模式，用于批处理模式（默认: {pdb_id}_segment.mrc）")
    parser.add_argument("--auto-reference", action="store_true", default=True, help="自动查找参考密度图（默认: True）")
    
    args = parser.parse_args()
    
    # 批处理模式
    if args.batch:
        if not args.data_root:
            print("错误: 批处理模式需要指定数据根目录 (-d/--data-root)")
            return 1
            
        if not args.protein_info:
            print("错误: 批处理模式需要指定蛋白质信息文件 (-p/--protein-info)")
            return 1
        
        # 处理数据目录
        process_data_directory(args.data_root, args.protein_info, args.segment_pattern)
        return 0
    
    # 单文件处理模式
    if not args.pdb:
        print("错误: 必须指定输入PDB文件路径 (--pdb)")
        return 1
    
    # 检查PDB文件是否存在
    if not os.path.exists(args.pdb):
        # 尝试查找其他可能的名称模式
        pdb_dir = os.path.dirname(args.pdb)
        pdb_basename = os.path.basename(args.pdb)
        pdb_id_match = re.match(r'([^\.]+)\.pdb', pdb_basename)
        
        if pdb_id_match and pdb_dir:
            pdb_id = pdb_id_match.group(1)
            # 尝试不同的文件名模式
            possible_patterns = [
                os.path.join(pdb_dir, f"{pdb_id}*.pdb"),
                os.path.join(pdb_dir, f"*{pdb_id}*.pdb"),
                os.path.join(pdb_dir, f"PDB-{pdb_id}.pdb"),
                os.path.join(pdb_dir, "*.pdb")
            ]
            
            for pattern in possible_patterns:
                matching_files = glob.glob(pattern)
                if matching_files:
                    args.pdb = matching_files[0]
                    print(f"找到替代PDB文件: {args.pdb}")
                    break
        
        if not os.path.exists(args.pdb):
            print(f"错误: 找不到PDB文件: {args.pdb}")
            return 1
        
    if not args.output:
        # 自动确定输出路径
        pdb_dir = os.path.dirname(args.pdb)
        # 提取基础PDB ID（不带PDB-前缀）
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.pdb', pdb_basename, re.IGNORECASE)
        
        if pdb_name_match:
            pdb_name = pdb_name_match.group(1).lower()
        else:
            pdb_name = os.path.splitext(pdb_basename)[0].lower()
            
        processed_dir = os.path.join(pdb_dir, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        args.output = os.path.join(processed_dir, f"{pdb_name}_backbone.mrc")
        print(f"未指定输出路径，将使用: {args.output}")
        
    # 如果提供了蛋白质信息文件和参考密度图，尝试获取蛋白质信息
    contour_level = args.contour
    resolution = args.resolution
    
    if args.protein_info:
        # 尝试从PDB文件路径或文件名中提取蛋白质ID
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.pdb', pdb_basename, re.IGNORECASE)
        
        if pdb_name_match:
            pdb_id = pdb_name_match.group(1).lower()
            
            # 查找目录名中的EMD ID
            parent_dir = os.path.basename(os.path.dirname(args.pdb))
            emd_match = re.search(r'EMD-(\d+)', parent_dir)
            
            if emd_match:
                emd_id = emd_match.group(0)
                
                # 解析蛋白质信息文件
                protein_info = parse_protein_info_file(args.protein_info)
                info = get_protein_info_by_emd_id(protein_info, emd_id)
                
                if info:
                    print(f"找到EMD ID {emd_id} 的信息:")
                    print(f"分辨率: {info['resolution']}")
                    print(f"等值面水平: {info['contour_level']}")
                    
                    resolution = info['resolution']
                    contour_level = info['contour_level']
    
    # 自动查找参考密度图
    reference_map = args.reference_map
    if not reference_map and args.auto_reference:
        # 获取PDB文件信息
        pdb_dir = os.path.dirname(args.pdb)
        # 提取基础PDB ID（不带PDB-前缀）
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.pdb', pdb_basename, re.IGNORECASE)
        
        if pdb_name_match:
            pdb_name = pdb_name_match.group(1).lower()
        else:
            pdb_name = os.path.splitext(pdb_basename)[0].lower()
            
        # 查找可能的参考密度图
        possible_locations = [
            # 1. 同级目录中的segment.mrc文件
            os.path.join(pdb_dir, f"{pdb_name}_segment.mrc"),
            # 2. processed子目录中的segment.mrc文件
            os.path.join(pdb_dir, "processed", f"{pdb_name}_segment.mrc"),
            # 3. 同级目录中的任何.mrc文件（排除backbone.mrc）
            *[f for f in glob.glob(os.path.join(pdb_dir, f"{pdb_name}_*.mrc")) 
              if not f.endswith("_backbone.mrc")],
            # 4. processed子目录中的任何.mrc文件（排除backbone.mrc）
            *[f for f in glob.glob(os.path.join(pdb_dir, "processed", f"{pdb_name}_*.mrc")) 
              if not f.endswith("_backbone.mrc")]
        ]
        
        # 查找第一个存在的参考密度图
        for loc in possible_locations:
            if os.path.exists(loc):
                reference_map = loc
                print(f"自动找到参考密度图: {reference_map}")
                break
                
        if not reference_map:
            print("警告: 未找到参考密度图，将使用默认参数生成骨架密度图")
    
    # 生成骨架密度图
    generate_backbone_density(
        input_pdb=args.pdb,
        output_mrc=args.output,
        resolution=resolution,
        backbone_only=args.backbone_only,
        normalize=args.normalize,
        contour_level=contour_level,
        reference_map=reference_map
    )
    print(f"成功: 骨架密度图已保存到 '{args.output}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
