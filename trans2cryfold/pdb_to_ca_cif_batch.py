#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from collections import defaultdict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量将PDB文件转换为CryFold可用的CIF格式")
    parser.add_argument("--input_dir", required=True, help="输入目录，包含多个蛋白质子目录")
    parser.add_argument("--protein_list", required=True, help="蛋白质名称列表文件，每行一个")
    parser.add_argument("--only_ca", action="store_true", help="是否只提取CA原子")
    return parser.parse_args()

def load_pdb(pdb_file, only_ca=True):
    """
    加载PDB文件并提取原子坐标
    
    参数:
    - pdb_file: PDB文件路径
    - only_ca: 是否只提取CA原子
    
    返回:
    - atom_coords: 原子坐标数组，shape为(n_atoms, 3)
    - atom_names: 原子名称列表
    - residue_names: 残基名称列表
    - residue_ids: 残基ID列表
    """
    # 定义要提取的原子
    atom_coords = []
    atom_names = []
    residue_names = []
    residue_ids = []
    
    # 用于检测重复残基
    seen_residues = defaultdict(list)
    
    # 根据文件扩展名选择适当的解析器
    if pdb_file.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
        print(f"使用MMCIFParser解析CIF文件: {pdb_file}")
    else:
        parser = PDBParser(QUIET=True)
        print(f"使用PDBParser解析PDB文件: {pdb_file}")
    
    # 忽略PDB构建警告    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        
        try:
            structure = parser.get_structure('structure', pdb_file)
            
            for model in structure:
                for chain in model:
                    # 建立残基字典，处理重复的残基
                    processed_residues = {}
                    for residue in chain:
                        # 跳过非标准残基
                        if residue.id[0] != ' ':
                            continue
                        
                        # 使用残基ID作为键
                        res_key = residue.id[1]
                        if res_key in processed_residues:
                            # 如果残基ID已存在，检查是否包含CA原子
                            existing_has_ca = any(atom.get_name() == "CA" for atom in processed_residues[res_key])
                            current_has_ca = any(atom.get_name() == "CA" for atom in residue)
                            
                            # 如果当前残基有CA而已存在的没有，则替换
                            if current_has_ca and not existing_has_ca:
                                processed_residues[res_key] = residue
                                print(f"警告: 残基ID {res_key} 重复定义，使用包含CA原子的那个")
                        else:
                            processed_residues[res_key] = residue
                    
                    # 处理过滤后的残基
                    for res_id, residue in sorted(processed_residues.items()):
                        for atom in residue:
                            # 获取原子名称
                            atom_name = atom.get_name()
                            
                            # 如果只提取CA原子，则跳过其他原子
                            if only_ca and atom_name != "CA":
                                continue
                                
                            # 检查原子元素是否为未知元素'X'，如果是则设为'C'
                            if atom.element == 'X' or atom.element == '':
                                atom.element = 'C'  # 将未知元素设为碳原子
                                
                            atom_coords.append(atom.get_coord())
                            atom_names.append(atom_name)
                            residue_names.append(residue.get_resname())
                            residue_ids.append(residue.id[1])  # 获取残基编号
            
            print(f"从结构文件中提取了 {len(atom_coords)} 个{'CA' if only_ca else ''}原子")
            return atom_coords, atom_names, residue_names, residue_ids
        except Exception as e:
            print(f"解析文件 {pdb_file} 时出错: {e}")
            return [], [], [], []

def points_to_cif(path_to_save, coords, atom_names, residue_names, residue_ids):
    """
    将坐标点保存为CIF格式，使用与CryFold完全相同的方法
    
    参数:
    - path_to_save: 输出文件路径
    - coords: 原子坐标数组 (N x 3)
    - atom_names: 原子名称列表
    - residue_names: 残基名称列表
    - residue_ids: 残基ID列表
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    
    # 创建结构
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")  # 使用链ID "1"
    
    prev_res_id = None
    added_atoms = set()  # 跟踪已添加的原子，避免重复
    
    # 先对输入数据进行处理，确保相同残基ID只出现一次
    unique_residues = {}
    for i, (point, atom_name, res_name, res_id) in enumerate(zip(coords, atom_names, residue_names, residue_ids)):
        if res_id not in unique_residues:
            unique_residues[res_id] = {'coords': [], 'atom_names': [], 'res_name': res_name}
        
        # 检查是否已存在相同的原子
        atom_exists = False
        for existing_atom_name in unique_residues[res_id]['atom_names']:
            if existing_atom_name == atom_name:
                atom_exists = True
                break
                
        if not atom_exists:
            unique_residues[res_id]['coords'].append(point)
            unique_residues[res_id]['atom_names'].append(atom_name)
    
    # 按残基ID排序
    for res_id in sorted(unique_residues.keys()):
        res_data = unique_residues[res_id]
        res_name = res_data['res_name']
        
        # 初始化残基
        struct.init_residue(res_name, " ", res_id, " ")
        
        # 添加该残基的所有原子
        for point, atom_name in zip(res_data['coords'], res_data['atom_names']):
            # 确保原子元素不为空或'X'
            element = atom_name[0] if atom_name[0] not in ['X', ' '] else 'C'
            
            # 设置原子
            struct.init_atom(atom_name, point, 0, 1, " ", atom_name, element)
    
    # 获取构建的结构
    structure = struct.get_structure()
    
    # 保存为CIF格式
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(path_to_save)
    
    print(f"成功保存CIF文件到: {path_to_save}")

def convert_pdb_to_ca_cif(input_pdb, output_cif, only_ca=True):
    """
    将PDB文件转换为CryFold兼容的CIF文件
    
    参数:
    - input_pdb: 输入PDB文件路径
    - output_cif: 输出CIF文件路径
    - only_ca: 是否只提取CA原子
    
    返回:
    - output_cif: 输出CIF文件路径
    """
    # 加载PDB文件并提取原子坐标
    atom_coords, atom_names, residue_names, residue_ids = load_pdb(input_pdb, only_ca)
    
    if len(atom_coords) == 0:
        print(f"警告: 从文件 {input_pdb} 中未能提取到原子坐标！")
        return None
    
    # 将坐标点转换为CIF格式并保存
    points_to_cif(output_cif, atom_coords, atom_names, residue_names, residue_ids)
    
    return output_cif

def batch_process(input_dir, protein_list_file, only_ca=True):
    """
    批量处理PDB文件，转换为CIF格式
    
    参数:
    - input_dir: 输入目录，包含多个蛋白质子目录
    - protein_list_file: 蛋白质名称列表文件，每行一个
    - only_ca: 是否只提取CA原子
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
                    
                    # 按照优先级查找PDB文件和CIF文件
                    potential_files = [
                        os.path.join(pdb_dir, f"{pdb_code}.pdb"),
                        os.path.join(pdb_dir, f"PDB-{pdb_code}.pdb"),
                        os.path.join(pdb_dir, dir_name + ".pdb"),
                        # 添加CIF文件的查找
                        os.path.join(pdb_dir, f"{pdb_code}.cif"),
                        os.path.join(pdb_dir, f"PDB-{pdb_code}.cif"),
                        os.path.join(pdb_dir, dir_name + ".cif")
                    ]
                    
                    for potential_file in potential_files:
                        if os.path.exists(potential_file):
                            pdb_file = potential_file
                            break
                    
                    # 如果没有找到标准命名的文件，则搜索文件夹中的所有PDB和CIF文件
                    if pdb_file is None:
                        for potential_file in os.listdir(pdb_dir):
                            if potential_file.lower().endswith(('.pdb', '.cif')):
                                pdb_file = os.path.join(pdb_dir, potential_file)
                                break
                    
                    # 特殊情况：如果仍然找不到文件，尝试使用PDB-7rdr.cif文件
                    if pdb_file is None and os.path.exists(os.path.join(pdb_dir, "PDB-7rdr.cif")):
                        pdb_file = os.path.join(pdb_dir, "PDB-7rdr.cif")
                    
                    if pdb_file is not None:
                        # 定义输出CIF文件路径
                        output_cif = os.path.join(pdb_dir, f"{pdb_code}_point.cif")
                        
                        print(f"  找到结构文件: {pdb_file}")
                        print(f"  将转换为CIF文件: {output_cif}")
                        
                        try:
                            # 转换PDB为CIF
                            result = convert_pdb_to_ca_cif(pdb_file, output_cif, only_ca)
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
                        print(f"  错误: 在文件夹 {pdb_dir} 中找不到PDB或CIF文件")
    
    print(f"\n批处理完成！成功转换: {processed_count}, 失败: {error_count}")

def main():
    args = parse_args()
    batch_process(args.input_dir, args.protein_list, args.only_ca)

if __name__ == "__main__":
    main()
