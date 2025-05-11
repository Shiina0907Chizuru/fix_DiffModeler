#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import re
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="提取指定蛋白质的segment.mrc文件")
    parser.add_argument("--input_dir", required=True, help="包含蛋白质子文件夹的主目录")
    parser.add_argument("--output_dir", required=True, help="segment.mrc文件的输出目录")
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
                
                # 提取格式为"ID: 值"的ID部分
                parts = line.split(':', 1)
                if len(parts) >= 1:
                    pdb_id = parts[0].strip().lower()
                    pdb_ids.append(pdb_id)
    except Exception as e:
        print(f"读取信息文件时出错: {str(e)}")
    
    return pdb_ids

def find_segment_files(input_dir, pdb_ids):
    """
    在目录中查找指定PDB ID的segment.mrc文件
    
    参数:
    - input_dir: 输入目录
    - pdb_ids: 需要处理的蛋白质ID列表
    
    返回:
    - segment_files: segment.mrc文件路径列表及对应的PDB ID
    """
    segment_files = []
    
    # 将PDB ID列表转换为set，提高查找效率
    pdb_id_set = set(pdb_ids)
    
    # 查找所有可能的蛋白质文件夹
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        # 确保是目录
        if not os.path.isdir(item_path):
            continue
        
        # 检查目录名称是否包含PDB ID
        folder_pdb_id = None
        for pdb_id in pdb_id_set:
            if pdb_id.lower() in item.lower():
                folder_pdb_id = pdb_id
                break
        
        if folder_pdb_id is None:
            continue
        
        # 查找processed子文件夹
        processed_dir = os.path.join(item_path, "processed")
        if not os.path.isdir(processed_dir):
            print(f"在{item_path}中未找到processed文件夹，尝试直接查找segment文件")
            processed_dir = item_path
        
        # 在processed文件夹中查找segment.mrc文件
        segment_pattern = re.compile(r'.*segment\.mrc$', re.IGNORECASE)
        for file in os.listdir(processed_dir):
            if segment_pattern.match(file):
                segment_path = os.path.join(processed_dir, file)
                segment_files.append((segment_path, folder_pdb_id))
                print(f"找到segment文件: {segment_path}")
                break
    
    return segment_files

def extract_segments(input_dir, output_dir, info_file, copy_mode="copy"):
    """
    提取segment.mrc文件
    
    参数:
    - input_dir: 输入目录
    - output_dir: 输出目录
    - info_file: 信息文件路径
    - copy_mode: 复制模式("copy"或"move")
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取信息文件，获取需要处理的蛋白质ID
    pdb_ids = read_info_file(info_file)
    print(f"从信息文件中读取了{len(pdb_ids)}个PDB ID")
    
    # 在目录中查找指定PDB ID的segment.mrc文件
    segment_files = find_segment_files(input_dir, pdb_ids)
    print(f"找到{len(segment_files)}个segment.mrc文件")
    
    # 复制或移动文件到输出目录
    for segment_path, pdb_id in segment_files:
        # 构建输出文件名
        output_filename = f"{pdb_id}_segment.mrc"
        output_path = os.path.join(output_dir, output_filename)
        
        # 复制或移动文件
        try:
            if copy_mode == "copy":
                shutil.copy2(segment_path, output_path)
                print(f"已复制: {segment_path} -> {output_path}")
            else:  # move
                shutil.move(segment_path, output_path)
                print(f"已移动: {segment_path} -> {output_path}")
        except Exception as e:
            print(f"处理文件{segment_path}时出错: {str(e)}")
    
    print("\n提取完成!")
    print(f"总共处理了{len(segment_files)}/{len(pdb_ids)}个文件")

def main():
    args = parse_args()
    extract_segments(args.input_dir, args.output_dir, args.info_file, args.copy_mode)

if __name__ == "__main__":
    main()
