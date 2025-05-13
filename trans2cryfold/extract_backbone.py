#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import re
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="提取指定蛋白质的backbone.mrc文件")
    parser.add_argument("--input_dir", required=True, help="包含蛋白质子文件夹的主目录")
    parser.add_argument("--output_dir", required=True, help="backbone.mrc文件的输出目录")
    parser.add_argument("--info_file", required=True, help="包含需要提取的蛋白质ID信息的文件")
    parser.add_argument("--copy_mode", choices=["copy", "move"], default="copy", 
                        help="是复制文件还是移动文件，默认为复制")
    return parser.parse_args()

def read_info_file(file_path):
    """
    读取信息文件，获取需要处理的蛋白质ID
    
    参数:
    - file_path: 信息文件路径
    
    返回:
    - pdb_ids: 需要处理的蛋白质ID列表
    """
    pdb_ids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 提取格式为"ID: 值"的ID部分或直接是ID
                parts = line.split(':', 1)
                pdb_id = parts[0].strip().lower()
                pdb_ids.append(pdb_id)
    except Exception as e:
        print(f"读取信息文件时出错: {str(e)}")
    
    return pdb_ids

def find_backbone_files(input_dir, pdb_ids):
    """
    在目录中查找指定PDB ID的backbone.mrc文件
    
    参数:
    - input_dir: 输入目录
    - pdb_ids: 需要处理的蛋白质ID列表
    
    返回:
    - backbone_files: 找到的backbone.mrc文件路径字典，键为PDB ID，值为文件路径
    """
    backbone_files = {}
    not_found = []
    
    # 遍历主目录下的所有子目录
    for root, dirs, files in os.walk(input_dir):
        # 检查是否是PDB-{pdb_id}-EMD-xxxxx/processed目录
        match = re.search(r'PDB-(\w+)-EMD-\d+[/\\]processed', root)
        if match:
            pdb_id = match.group(1).lower()
            if pdb_id in pdb_ids:
                # 在processed目录中查找backbone.mrc文件
                for file in files:
                    if file.lower() == f"{pdb_id}_backbone.mrc" or file.lower() == f"{pdb_id.lower()}_backbone.mrc":
                        backbone_files[pdb_id] = os.path.join(root, file)
                        print(f"找到 {pdb_id} 的backbone.mrc文件: {os.path.join(root, file)}")
                        break
    
    # 检查哪些蛋白质没有找到backbone.mrc文件
    for pdb_id in pdb_ids:
        if pdb_id not in backbone_files:
            not_found.append(pdb_id)
    
    if not_found:
        print(f"警告: 未找到以下 {len(not_found)} 个蛋白质的backbone.mrc文件:")
        for pdb_id in not_found[:10]:  # 只打印前10个
            print(f"  {pdb_id}")
        if len(not_found) > 10:
            print(f"  ...以及其他 {len(not_found)-10} 个")
    
    return backbone_files

def extract_files(backbone_files, output_dir, copy_mode):
    """
    提取backbone.mrc文件到输出目录
    
    参数:
    - backbone_files: 找到的backbone.mrc文件路径字典
    - output_dir: 输出目录
    - copy_mode: 复制模式，"copy"或"move"
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for pdb_id, file_path in backbone_files.items():
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        
        try:
            if os.path.exists(output_file):
                print(f"警告: 目标文件 {output_file} 已存在，将被覆盖")
            
            if copy_mode == "copy":
                shutil.copy2(file_path, output_file)
                print(f"已复制: {file_path} -> {output_file}")
            else:  # move
                shutil.move(file_path, output_file)
                print(f"已移动: {file_path} -> {output_file}")
            
            success_count += 1
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return success_count

def main():
    """主函数"""
    args = parse_args()
    
    # 读取信息文件
    print(f"正在读取信息文件: {args.info_file}")
    pdb_ids = read_info_file(args.info_file)
    print(f"从信息文件中读取了 {len(pdb_ids)} 个蛋白质ID")
    
    if not pdb_ids:
        print("错误: 未从信息文件中读取到任何蛋白质ID")
        return
    
    # 查找backbone.mrc文件
    print(f"正在目录 {args.input_dir} 中查找backbone.mrc文件...")
    backbone_files = find_backbone_files(args.input_dir, pdb_ids)
    
    found_count = len(backbone_files)
    print(f"找到 {found_count}/{len(pdb_ids)} 个蛋白质的backbone.mrc文件")
    
    if not backbone_files:
        print("错误: 未找到任何backbone.mrc文件")
        return
    
    # 提取文件
    print(f"正在{args.copy_mode}文件到目录: {args.output_dir}")
    success_count = extract_files(backbone_files, args.output_dir, args.copy_mode)
    
    print(f"操作完成，成功处理了 {success_count}/{found_count} 个文件")

if __name__ == "__main__":
    main()
