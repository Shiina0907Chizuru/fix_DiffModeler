#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成骨架密度图工具

这个脚本用于从PDB或CIF文件生成蛋白质骨架的密度图。
它首先提取文件中的骨架原子（CA、C、N），然后将其转换为体积密度图。
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

# 直接定义filter_backbone函数，支持PDB和CIF格式
def filter_backbone(input_path, backbone_output_path):
    """
    提取PDB或CIF文件中的骨架原子
    
    Args:
        input_path (str): 输入PDB或CIF文件路径
        backbone_output_path (str): 输出只包含骨架原子的文件路径
    """
    # 定义骨架原子列表
    backbone_list = ["CA", "C", "N"]
    # 支持DNA/RNA模板拟合
    # backbone_list_drna = ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
    # backbone_list += backbone_list_drna
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # 使用BioPython处理PDB和CIF文件
    if file_ext == '.cif':
        try:
            # 方法1：使用Bio.PDB模块直接处理CIF文件(可能遇到解析错误)
            try:
                from Bio.PDB import MMCIFParser, PDBIO
                parser = MMCIFParser(QUIET=True)
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
                print(f"已从CIF文件提取骨架原子，保存为PDB格式: {backbone_output_path}")
                
            except Exception as e:
                print(f"使用BioPython处理CIF文件时出错: {e}")
                print("尝试使用备用方法处理CIF文件...")
                
                # 方法2：如果BioPython解析失败，尝试使用gemmi库处理CIF文件
                try:
                    import gemmi
                    structure = gemmi.read_structure(input_path)
                    
                    # 创建一个新的PDB文件
                    with open(backbone_output_path, 'w', errors='ignore') as out_file:
                        # 遍历所有模型、链、残基和原子
                        for model in structure:
                            for chain in model:
                                for residue in chain:
                                    for atom in residue:
                                        atom_name = atom.name.strip()
                                        if atom_name in backbone_list:
                                            # 格式化PDB ATOM记录
                                            x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                                            
                                            # 修复gemmi.Element格式化问题
                                            # 将gemmi.Element对象转换为字符串
                                            element_str = str(atom.element.name if hasattr(atom.element, 'name') else atom.element)
                                            if not element_str or len(element_str) > 2:
                                                # 如果没有有效的元素符号，根据原子名称推断
                                                element_str = atom_name[0] if atom_name else "C"
                                                
                                            # 确保元素字符串适合PDB格式
                                            element_str = element_str[:2].strip()
                                            
                                            # 处理residue.seqid.num可能不是数字的情况
                                            try:
                                                seq_num = int(residue.seqid.num)
                                                seq_num_str = f"{seq_num:4d}"
                                            except (ValueError, TypeError):
                                                # 如果无法转换为整数，就直接使用字符串
                                                seq_num_str = f"{str(residue.seqid.num)[:4]:4s}"
                                                
                                            # 确保没有额外的空格插入到坐标数据中
                                            try:
                                                line = f"ATOM  {atom.serial:5d} {atom.name:^4s} {residue.name:3s} {chain.name:1s}{seq_num_str}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {element_str:>2s}  \n"
                                            except Exception as format_error:
                                                print(f"格式化PDB行时出错: {format_error}")
                                                line = f"ATOM  {atom.serial % 100000:5d} {atom_name:^4s} {residue.name[:3]:3s} {chain.name[0]:1s}{seq_num_str}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {element_str:>2s}  \n"
                                            
                                            out_file.write(line)
                    
                    print(f"已使用gemmi库从CIF文件提取骨架原子: {backbone_output_path}")
                    
                except ImportError:
                    print("未安装gemmi库，尝试使用文本解析方法...")
                    
                    # 方法3：最后尝试直接解析CIF文本文件
                    try:
                        import re
                        atom_data = []
                        
                        with open(input_path, 'r', errors='ignore') as infile:
                            cif_content = infile.read()
                            
                        # 查找atom_site部分
                        atom_site_loop = re.search(r'_atom_site\..*?(?=_|\Z)', cif_content, re.DOTALL)
                        if atom_site_loop:
                            atom_loop = atom_site_loop.group(0)
                            
                            # 查找列名称
                            headers = re.findall(r'_atom_site\.(\S+)', atom_loop)
                            
                            # 找出关键列的索引
                            group_pdb_idx = -1
                            label_atom_id_idx = -1
                            
                            for i, header in enumerate(headers):
                                if header == 'group_PDB':
                                    group_pdb_idx = i
                                elif header == 'label_atom_id':
                                    label_atom_id_idx = i
                            
                            # 如果找到必要的列，解析原子数据
                            if group_pdb_idx != -1 and label_atom_id_idx != -1:
                                # 提取数据行
                                data_lines = re.findall(r'\n(\S+(?:\s+\S+)*)', atom_loop)
                                
                                with open(backbone_output_path, 'w', errors='ignore') as outfile:
                                    for line in data_lines:
                                        parts = line.split()
                                        if len(parts) > max(group_pdb_idx, label_atom_id_idx):
                                            if parts[group_pdb_idx] == 'ATOM':
                                                atom_name = parts[label_atom_id_idx].strip('"\'')
                                                if atom_name in backbone_list:
                                                    # 重新格式化为PDB ATOM记录
                                                    formatted_line = format_atom_line_from_cif(parts, headers)
                                                    if formatted_line:
                                                        outfile.write(formatted_line + '\n')
                        
                        print(f"已通过文本解析从CIF文件提取骨架原子: {backbone_output_path}")
                        
                    except Exception as e:
                        print(f"所有CIF处理方法均失败: {e}")
                        # 创建空的PDB文件以避免后续处理错误
                        with open(backbone_output_path, 'w', errors='ignore') as f:
                            f.write("REMARK Failed to process CIF file\n")
                        raise ValueError(f"无法处理CIF文件 {input_path}: {e}")
                
        except Exception as e:
            print(f"处理CIF文件时出错: {e}")
            raise
    else:
        # 原始的PDB文件处理方式，直接解析文本行
        try:
            with open(input_path, 'r', errors='ignore') as file:
                with open(backbone_output_path, 'w', errors='ignore') as wfile:
                    for line in file:
                        if line.startswith("ATOM"):
                            chain_name = line[21]
                            atom_name = line[12:16].replace(" ", "")
                            if atom_name in backbone_list:
                                wfile.write(line)
            print(f"已从PDB文件提取骨架原子: {backbone_output_path}")
        except Exception as e:
            print(f"处理PDB文件时出错: {e}")
            raise

def format_atom_line_from_cif(parts, headers):
    """从CIF数据行格式化PDB ATOM行"""
    try:
        # 查找必要字段的索引
        indices = {}
        for field in ['label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id', 
                     'Cartn_x', 'Cartn_y', 'Cartn_z', 'type_symbol']:
            try:
                indices[field] = headers.index(field)
            except ValueError:
                indices[field] = -1  # 字段不存在
        
        # 确保所有必要字段都存在
        required_fields = ['label_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z']
        if any(indices[field] == -1 for field in required_fields):
            return None
        
        # 提取数据
        atom_name = parts[indices['label_atom_id']].strip('"\'')
        
        # 获取残基名称(如果存在)
        res_name = "UNK"
        if indices['label_comp_id'] != -1 and len(parts) > indices['label_comp_id']:
            res_name = parts[indices['label_comp_id']].strip('"\'')
            
        # 获取链ID(如果存在)
        chain_id = "A"
        if indices['label_asym_id'] != -1 and len(parts) > indices['label_asym_id']:
            chain_id = parts[indices['label_asym_id']].strip('"\'')
            
        # 获取残基序号(如果存在)
        res_seq = "1"
        if indices['label_seq_id'] != -1 and len(parts) > indices['label_seq_id']:
            res_seq = parts[indices['label_seq_id']].strip('"\'')
            
        # 获取原子坐标
        x = float(parts[indices['Cartn_x']])
        y = float(parts[indices['Cartn_y']])
        z = float(parts[indices['Cartn_z']])
        
        # 获取元素类型(如果存在)
        element = " "
        if indices['type_symbol'] != -1 and len(parts) > indices['type_symbol']:
            element = parts[indices['type_symbol']].strip('"\'')
            
        # 格式化PDB ATOM行
        atom_serial = 1  # 默认序号
        occupancy = 1.0
        temp_factor = 0.0
        
        return f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:3s} {chain_id:1s}{res_seq:4s}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}"
        
    except Exception as e:
        print(f"格式化CIF原子行时出错: {e}")
        return None

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


def generate_backbone_density(input_file, output_mrc, resolution=1.0, backbone_only=True, normalize=True, contour_level=None, reference_map=None):
    """
    从PDB或CIF文件生成骨架密度图

    Args:
        input_file (str): 输入PDB或CIF文件路径
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
    
    # 创建临时目录存储骨架原子文件
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 判断文件类型
    file_ext = os.path.splitext(input_file)[1].lower()
    is_cif = file_ext == '.cif'
    
    # 创建临时PDB文件存储骨架原子
    base_name = Path(input_file).stem
    backbone_output = os.path.join(temp_dir, f"{base_name}_backbone.pdb")
    
    # 提取骨架原子
    filter_backbone(input_file, backbone_output)
    print(f"已提取骨架原子到: {backbone_output}")
    
    # 转换为体积密度图
    pdb2vol(
        input_pdb=backbone_output,
        resolution=resolution,
        output_mrc=output_mrc,
        normalize=normalize,
        backbone_only=backbone_only,
        contour=None,  # 设置为None，避免执行contour操作
        ref_map=reference_map  # 使用参考密度图保持尺寸一致
    )
    print(f"已生成骨架密度图: {output_mrc}")
    
    # 显示密度图信息
    try:
        print(f"尝试读取生成的MRC文件: {output_mrc}")
        with mrcfile.open(output_mrc) as mrc:
            data = mrc.data
            min_val = np.min(data)
            max_val = np.max(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            print(f"密度图大小: {data.shape}")
            print(f"密度值范围: {min_val:.6f} to {max_val:.6f}")
            print(f"密度图统计信息: 均值={mean_val:.6f}, 标准差={std_val:.6f}")
            print(f"检查负值: {'有' if min_val < 0 else '无'}负值")
            print(f"MRC文件模式: {mrc.header.mode}")
            
            # 计算正负值比例
            if min_val < 0:
                positive_pct = np.sum(data > 0) / data.size * 100
                negative_pct = np.sum(data < 0) / data.size * 100
                print(f"正值占比: {positive_pct:.2f}%, 负值占比: {negative_pct:.2f}%")
            
            # 手动验证是否存在负值
            has_negative = np.any(data < -0.001)  # 使用一个小阈值避免精度问题
            print(f"文件中是否存在负值: {has_negative}")
            
            if not has_negative and min_val >= 0 and mean_val > 0.1:
                print("警告: MRC文件可能未保存负值，请检查write_mrc_file函数实现")
    except Exception as e:
        print(f"读取密度图信息时出错: {e}")
    
    return output_mrc


def process_data_directory(data_root, protein_info_file, segment_pattern="{pdb_id}_segment.mrc", force_regenerate=False):
    """
    批量处理数据目录中的所有蛋白质，生成骨架密度图。
    
    每个蛋白质目录应该有格式：PDB-{pdb_id}-EMD-{emd_id}
    
    参数:
    - data_root (str): 数据根目录，包含所有蛋白质子目录
    - protein_info_file (str): 包含蛋白质信息（EMD ID、轮廓级别和分辨率）的文本文件
    - segment_pattern (str): 参考密度图文件名模式
    - force_regenerate (bool): 是否强制重新生成骨架密度图，即使输出文件已存在
    """
    # 解析蛋白质信息文件
    protein_info = parse_protein_info_file(protein_info_file)
    
    # 获取数据根目录下的所有PDB子目录
    pdb_dirs = glob.glob(os.path.join(data_root, "PDB-*-EMD-*"))
    
    print(f"找到{len(pdb_dirs)}个蛋白质目录")
    
    # 处理每个蛋白质目录
    success_count = 0
    skip_count = 0
    error_count = 0
    failed_proteins = []  # 记录失败的蛋白质名称
    
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
            
            # 创建处理后的输出目录
            processed_dir = os.path.join(pdb_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # 设置输出文件路径
            output_path = os.path.join(processed_dir, f"{pdb_id}_backbone.mrc")
            
            # 改进的检测逻辑: 检查processed文件夹中是否存在{pdb_id}_backbone.mrc文件
            if not force_regenerate:
                # 检查输出文件是否已存在
                if os.path.exists(output_path):
                    print(f"跳过 {pdb_id} (EMD-{emd_id})：骨架密度图已存在")
                    skip_count += 1
                    continue
                    
                # 检查processed目录中是否存在任何*_backbone.mrc文件
                backbone_files = glob.glob(os.path.join(processed_dir, "*_backbone.mrc"))
                if backbone_files:
                    print(f"跳过 {pdb_id} (EMD-{emd_id})：在processed目录中找到已有骨架密度图: {os.path.basename(backbone_files[0])}")
                    skip_count += 1
                    continue
            
            # 查找pdb或cif文件 - 支持多种命名模式
            possible_patterns = [
                # PDB文件模式
                os.path.join(pdb_dir, f"{pdb_id}*.pdb"),      # 基本名称: 3j3r.pdb
                os.path.join(pdb_dir, f"*{pdb_id}*.pdb"),     # 任何位置包含ID: PDB-3j3r.pdb
                os.path.join(pdb_dir, f"PDB-{pdb_id}.pdb"),   # 显式前缀: PDB-3j3r.pdb
                
                # CIF文件模式
                os.path.join(pdb_dir, f"{pdb_id}*.cif"),      # 基本名称: 3j3r.cif
                os.path.join(pdb_dir, f"*{pdb_id}*.cif"),     # 任何位置包含ID: PDB-3j3r.cif
                os.path.join(pdb_dir, f"PDB-{pdb_id}.cif"),   # 显式前缀: PDB-3j3r.cif
                
                # 如果以上都没找到，尝试任何PDB或CIF文件
                os.path.join(pdb_dir, "*.pdb"),               # 任何PDB文件
                os.path.join(pdb_dir, "*.cif")                # 任何CIF文件
            ]
            
            structure_file = None
            file_type = None
            for pattern in possible_patterns:
                matching_files = glob.glob(pattern)
                if matching_files:
                    structure_file = matching_files[0]
                    file_type = "PDB" if structure_file.lower().endswith(".pdb") else "CIF"
                    print(f"找到{file_type}文件: {structure_file}")
                    break
                
            if not structure_file:
                print(f"警告: 找不到{pdb_id}的PDB或CIF文件，跳过")
                failed_proteins.append(f"{pdb_id} (EMD-{emd_id}): 找不到结构文件")
                error_count += 1
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
            try:
                generate_backbone_density(
                    input_file=structure_file,
                    output_mrc=output_path,
                    resolution=resolution,
                    backbone_only=True,
                    normalize=True,
                    contour_level=contour_level,
                    reference_map=reference_map
                )
                success_count += 1
                print(f"成功处理 {pdb_id} (EMD-{emd_id})")
            except Exception as e:
                error_message = f"生成骨架密度图失败: {str(e)}"
                print(f"处理 {pdb_id} (EMD-{emd_id}) 时出错: {error_message}")
                failed_proteins.append(f"{pdb_id} (EMD-{emd_id}): {error_message}")
                error_count += 1
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            error_message = f"处理目录失败: {str(e)}"
            print(f"处理目录 {pdb_dir} 时出错: {error_message}")
            # 尝试提取蛋白质ID进行记录
            try:
                dir_name = os.path.basename(pdb_dir)
                match = re.match(r"PDB-([^-]+)-EMD-(\d+)", dir_name)
                if match:
                    pdb_id = match.group(1).lower()
                    emd_id = match.group(2)
                    failed_proteins.append(f"{pdb_id} (EMD-{emd_id}): {error_message}")
                else:
                    failed_proteins.append(f"{dir_name}: {error_message}")
            except:
                failed_proteins.append(f"{pdb_dir}: {error_message}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n处理完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}")
    
    # 输出失败的蛋白质名称
    if failed_proteins:
        print("\n失败的蛋白质列表:")
        for i, protein in enumerate(failed_proteins, 1):
            print(f"{i}. {protein}")
    else:
        print("\n所有蛋白质处理成功或跳过")


def main():
    parser = argparse.ArgumentParser(description="从PDB或CIF文件生成蛋白质骨架密度图")
    parser.add_argument("--pdb", help="输入PDB或CIF文件路径")
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
    parser.add_argument("--force-regenerate", action="store_true", default=False, help="强制重新生成骨架密度图，即使输出文件已存在（默认: False）")
    
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
        process_data_directory(args.data_root, args.protein_info, args.segment_pattern, args.force_regenerate)
        return 0
    
    # 单文件处理模式
    if not args.pdb:
        print("错误: 必须指定输入PDB或CIF文件路径 (--pdb)")
        return 1
    
    # 检查PDB或CIF文件是否存在
    if not os.path.exists(args.pdb):
        # 尝试查找其他可能的名称模式
        pdb_dir = os.path.dirname(args.pdb)
        pdb_basename = os.path.basename(args.pdb)
        pdb_id_match = re.match(r'([^\.]+)\.(pdb|cif)', pdb_basename)
        
        if pdb_id_match and pdb_dir:
            pdb_id = pdb_id_match.group(1)
            # 尝试不同的文件名模式
            possible_patterns = [
                # PDB文件模式
                os.path.join(pdb_dir, f"{pdb_id}*.pdb"),
                os.path.join(pdb_dir, f"*{pdb_id}*.pdb"),
                os.path.join(pdb_dir, f"PDB-{pdb_id}.pdb"),
                os.path.join(pdb_dir, "*.pdb"),
                os.path.join(pdb_dir, f"{pdb_id}*.cif"),
                os.path.join(pdb_dir, f"*{pdb_id}*.cif"),
                os.path.join(pdb_dir, f"PDB-{pdb_id}.cif"),
                os.path.join(pdb_dir, "*.cif")
            ]
            
            for pattern in possible_patterns:
                pdb_files = glob.glob(pattern)
                if pdb_files:
                    args.pdb = pdb_files[0]
                    print(f"找到替代PDB或CIF文件: {args.pdb}")
                    break
        
        if not os.path.exists(args.pdb):
            print(f"错误: 找不到PDB或CIF文件: {args.pdb}")
            return 1
        
    if not args.output:
        # 自动确定输出路径
        pdb_dir = os.path.dirname(args.pdb)
        # 提取基础PDB ID（不带PDB-前缀）
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.(?:pdb|cif)', pdb_basename, re.IGNORECASE)
        
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
        # 尝试从PDB或CIF文件路径或文件名中提取蛋白质ID
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.(?:pdb|cif)', pdb_basename, re.IGNORECASE)
        
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
        # 获取PDB或CIF文件信息
        pdb_dir = os.path.dirname(args.pdb)
        # 提取基础PDB ID（不带PDB-前缀）
        pdb_basename = os.path.basename(args.pdb)
        pdb_name_match = re.match(r'(?:PDB-)?([^\.]+)\.(?:pdb|cif)', pdb_basename, re.IGNORECASE)
        
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
        input_file=args.pdb,
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
