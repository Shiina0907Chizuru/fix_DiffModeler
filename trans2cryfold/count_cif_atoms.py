#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob
from pathlib import Path
from collections import defaultdict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB构建警告
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="统计CIF文件中的原子数量")
    parser.add_argument("--input_dir", required=True, help="包含CIF文件的输入目录")
    parser.add_argument("--output_file", help="输出文件路径(默认为input_dir/atom_counts.txt)")
    parser.add_argument("--recursive", action="store_true", help="是否递归搜索子目录")
    return parser.parse_args()

def extract_pdb_id(file_path):
    """
    从文件路径提取PDB ID
    
    参数:
    - file_path: 文件路径
    
    返回:
    - pdb_id: PDB ID
    """
    file_name = os.path.basename(file_path)
    # 假设文件名格式为XXXX_point.cif，提取前缀作为PDB ID
    pdb_id = file_name.split('_')[0].lower()
    return pdb_id

def count_atoms_in_cif(cif_file):
    """
    计算CIF文件中的原子数量
    
    参数:
    - cif_file: CIF文件路径
    
    返回:
    - atom_count: 原子数量
    """
    try:
        parser = MMCIFParser()
        structure = parser.get_structure("temp", cif_file)
        
        # 计算所有原子的数量
        atom_count = sum(1 for _ in structure.get_atoms())
        
        # 也可以专门计算CA原子数量，视需求而定
        ca_count = sum(1 for atom in structure.get_atoms() if atom.get_name() == "CA")
        
        return ca_count  # 返回CA原子数量，因为这些文件主要包含CA原子
    except Exception as e:
        print(f"解析文件 {cif_file} 时出错: {str(e)}")
        return 0

def find_all_cif_files(input_dir, recursive=False):
    """
    查找所有CIF文件
    
    参数:
    - input_dir: 输入目录
    - recursive: 是否递归搜索子目录
    
    返回:
    - cif_files: CIF文件路径列表
    """
    if recursive:
        return glob.glob(os.path.join(input_dir, "**", "*.cif"), recursive=True)
    else:
        return glob.glob(os.path.join(input_dir, "*.cif"))

def process_cif_files(input_dir, output_file=None, recursive=False):
    """
    处理所有CIF文件并统计原子数量
    
    参数:
    - input_dir: 输入目录
    - output_file: 输出文件
    - recursive: 是否递归搜索子目录
    """
    # 如果未指定输出文件，使用默认路径
    if output_file is None:
        output_file = os.path.join(input_dir, "atom_counts.txt")
    
    # 查找所有CIF文件
    cif_files = find_all_cif_files(input_dir, recursive)
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    # 按PDB ID对文件进行排序
    cif_files_by_pdb = defaultdict(list)
    for cif_file in cif_files:
        pdb_id = extract_pdb_id(cif_file)
        cif_files_by_pdb[pdb_id].append(cif_file)
    
    # 统计每个文件的原子数量
    results = {}
    for pdb_id, files in cif_files_by_pdb.items():
        # 如果一个PDB ID有多个文件，选择第一个文件
        cif_file = files[0]
        atom_count = count_atoms_in_cif(cif_file)
        results[pdb_id] = atom_count
        print(f"{pdb_id}: {atom_count} CA原子")
    
    # 写入结果到文件
    with open(output_file, 'w') as f:
        f.write("# PDB ID: CA原子数量\n")
        for pdb_id, count in sorted(results.items()):
            f.write(f"{pdb_id}: {count}\n")
    
    print(f"结果已保存至: {output_file}")
    return results

def main():
    args = parse_args()
    process_cif_files(args.input_dir, args.output_file, args.recursive)

if __name__ == "__main__":
    main()
