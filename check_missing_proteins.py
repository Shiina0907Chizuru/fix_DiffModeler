#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查不在信息文件中的蛋白质

该脚本比较/zhaoxuanj/FinialPDB目录中的蛋白质与20250315contour_level.txt文件中列出的蛋白质，
找出在目录中存在但不在信息文件中的蛋白质。
"""

import os
import argparse

def read_protein_ids_from_info_file(info_file_path):
    """从信息文件中读取蛋白质ID列表"""
    protein_ids = set()
    
    try:
        with open(info_file_path, 'r') as f:
            for line in f:
                # 处理每一行，格式应该是"protein_id: contour_level"
                line = line.strip()
                if line:
                    # 分割并提取蛋白质ID部分
                    parts = line.split(':')
                    if parts:
                        protein_id = parts[0].strip().lower()  # 转为小写以确保匹配
                        protein_ids.add(protein_id)
        
        print(f"从信息文件中读取到 {len(protein_ids)} 个蛋白质ID")
        return protein_ids
    except Exception as e:
        print(f"读取信息文件时出错: {e}")
        return set()

def extract_protein_id_from_dir_name(dir_name):
    """从目录名中提取蛋白质ID
    
    目录名格式: PDB-{protein_id}-EMD-{emd_id}
    例如: PDB-5jzh-EMD-8185 -> 5jzh
    """
    if not dir_name.startswith("PDB-"):
        return None
    
    # 分割目录名并提取蛋白质ID部分
    parts = dir_name.split('-')
    if len(parts) >= 2:
        return parts[1].lower()  # 转为小写以确保匹配
    
    return None

def find_missing_proteins(data_root, info_file_path):
    """查找在目录中存在但不在信息文件中的蛋白质"""
    # 读取信息文件中的蛋白质ID
    info_file_proteins = read_protein_ids_from_info_file(info_file_path)
    
    # 读取目录中的蛋白质ID
    dir_proteins = {}  # 使用字典存储目录名和对应的蛋白质ID
    missing_proteins = []
    
    try:
        # 列出数据根目录中的所有项目
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                protein_id = extract_protein_id_from_dir_name(item)
                if protein_id:
                    dir_proteins[item] = protein_id
                    # 检查该蛋白质ID是否在信息文件中
                    if protein_id not in info_file_proteins:
                        missing_proteins.append((item, protein_id))
        
        print(f"在目录中发现 {len(dir_proteins)} 个蛋白质目录")
        print(f"发现 {len(missing_proteins)} 个不在信息文件中的蛋白质")
        
        # 打印缺失的蛋白质
        if missing_proteins:
            print("\n不在信息文件中的蛋白质:")
            for dir_name, protein_id in missing_proteins:
                print(f"  {dir_name} (ID: {protein_id})")
            
            # 将缺失的蛋白质保存到文件
            output_file = "missing_proteins.txt"
            with open(output_file, 'w') as f:
                f.write("# 不在信息文件中的蛋白质\n")
                f.write("# 目录名 (蛋白质ID)\n")
                for dir_name, protein_id in missing_proteins:
                    f.write(f"{dir_name} ({protein_id})\n")
            print(f"\n已将缺失的蛋白质列表保存到 {output_file}")
        
        return missing_proteins
    except Exception as e:
        print(f"查找缺失蛋白质时出错: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="检查不在信息文件中的蛋白质")
    parser.add_argument("--data-root", type=str, default="/zhaoxuanj/FinialPDB", 
                        help="数据根目录路径 (默认: /zhaoxuanj/FinialPDB)")
    parser.add_argument("--info-file", type=str, default="20250315contour_level.txt", 
                        help="信息文件路径 (默认: 20250315contour_level.txt)")
    
    args = parser.parse_args()
    
    print(f"检查缺失的蛋白质...")
    print(f"数据根目录: {args.data_root}")
    print(f"信息文件: {args.info_file}")
    
    find_missing_proteins(args.data_root, args.info_file)
    
    print("\n检查完成!")

if __name__ == "__main__":
    main()
