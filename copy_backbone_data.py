#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import shutil
import argparse
from tqdm import tqdm

def parse_protein_info_file(file_path):
    """
    解析蛋白质信息文件，提取蛋白质名称
    
    参数:
    - file_path: 信息文件路径
    
    返回:
    - proteins: 蛋白质名称集合
    """
    proteins = set()
    
    try:
        # 尝试不同的编码打开文件
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
                return set()
        
        if file_content is None:
            print("所有编码方式都无法读取文件")
            return set()
        
        for line in file_content:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 解析格式：蛋白质名称 覆盖率
            # 例如：6adq: 0.6389
            match = re.match(r'^(\w+):?\s+[\d\.]+', line)
            if match:
                protein_name = match.group(1).strip()
                proteins.add(protein_name)
                print(f"从信息文件中解析到蛋白质: {protein_name}")
            else:
                parts = line.split()
                if len(parts) >= 1:
                    protein_name = parts[0].strip().rstrip(':')
                    proteins.add(protein_name)
                    print(f"从信息文件中解析到蛋白质: {protein_name}")
    
    except Exception as e:
        print(f"解析信息文件时出错: {str(e)}")
    
    return proteins

def extract_protein_code(folder_name):
    """
    从文件夹名称中提取蛋白质代码
    
    参数:
    - folder_name: 文件夹名称，如 PDB-8ew0-EMD-28639
    
    返回:
    - protein_code: 蛋白质代码，如 8ew0
    """
    # 尝试多种模式匹配
    patterns = [
        r'PDB-(\w+)-EMD-\d+',  # PDB-8ew0-EMD-28639
        r'PDB-(\w+)-\w+-\d+',   # 其他可能的格式
        r'PDB-(\w+)',          # PDB-8ew0
    ]
    
    for pattern in patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # 如果没有匹配到模式，返回文件夹名称
    return folder_name.lower()

def copy_backbone_data(source_dir, target_dir, info_file):
    """
    复制蛋白质的backbone数据
    
    参数:
    - source_dir: 源目录，包含所有蛋白质文件夹
    - target_dir: 目标目录，将创建蛋白质子文件夹
    - info_file: 信息文件路径，包含要跳过的蛋白质
    
    返回:
    - processed_proteins: 处理的蛋白质列表
    """
    # 解析信息文件，获取要跳过的蛋白质
    skip_proteins = parse_protein_info_file(info_file)
    print(f"从信息文件中解析到 {len(skip_proteins)} 个蛋白质，这些蛋白质将被跳过")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 记录处理的蛋白质
    processed_proteins = []
    
    # 扫描源目录
    print(f"开始扫描目录: {source_dir}")
    for folder_name in tqdm(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        
        # 跳过非目录
        if not os.path.isdir(folder_path):
            continue
        
        # 提取蛋白质代码
        protein_code = extract_protein_code(folder_name)
        
        # 检查是否在跳过列表中
        if protein_code in skip_proteins:
            print(f"跳过蛋白质 {protein_code}，因为它在信息文件中")
            continue
        
        # 查找backbone_Dataset目录
        backbone_dir = os.path.join(folder_path, "backbone_Dataset")
        if not os.path.exists(backbone_dir):
            print(f"蛋白质 {protein_code} 的backbone_Dataset目录不存在")
            continue
        
        # 创建目标子目录
        protein_target_dir = os.path.join(target_dir, protein_code)
        os.makedirs(protein_target_dir, exist_ok=True)
        
        # 计数复制的文件数量
        copied_files = 0
        
        # 遍历backbone_Dataset目录，查找input和output的npy文件
        for root, dirs, files in os.walk(backbone_dir):
            for file in files:
                # 只复制input_*.npy和output_*.npy文件
                if file.startswith(("input_", "output_")) and file.endswith(".npy"):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(protein_target_dir, file)
                    
                    # 复制文件
                    try:
                        shutil.copy2(source_file, target_file)
                        copied_files += 1
                    except Exception as e:
                        print(f"复制文件 {source_file} 时出错: {str(e)}")
        
        if copied_files > 0:
            print(f"成功复制蛋白质 {protein_code} 的 {copied_files} 个文件")
            processed_proteins.append(protein_code)
        else:
            print(f"蛋白质 {protein_code} 没有找到可复制的文件")
            # 如果没有复制任何文件，删除创建的空目录
            try:
                os.rmdir(protein_target_dir)
            except:
                pass
    
    return processed_proteins

def main():
    parser = argparse.ArgumentParser(description='复制蛋白质的backbone数据')
    parser.add_argument('--source_dir', type=str, required=True, help='源目录，包含所有蛋白质文件夹')
    parser.add_argument('--target_dir', type=str, required=True, help='目标目录，将创建蛋白质子文件夹')
    parser.add_argument('--info_file', type=str, required=True, help='信息文件路径，包含要跳过的蛋白质')
    parser.add_argument('--output_txt', type=str, default='processed_proteins.txt', help='输出的处理蛋白质列表文件')
    
    args = parser.parse_args()
    
    print(f"开始处理，源目录: {args.source_dir}")
    print(f"目标目录: {args.target_dir}")
    print(f"信息文件: {args.info_file}")
    
    # 复制数据
    processed_proteins = copy_backbone_data(args.source_dir, args.target_dir, args.info_file)
    
    # 保存处理的蛋白质列表
    output_path = os.path.join(os.path.dirname(args.target_dir), args.output_txt)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 处理的蛋白质列表\n")
            for protein in processed_proteins:
                f.write(f"{protein}\n")
        print(f"已将处理的 {len(processed_proteins)} 个蛋白质名称保存至: {output_path}")
    except Exception as e:
        print(f"保存处理蛋白质列表时出错: {str(e)}")
    
    print("处理完成")

if __name__ == "__main__":
    main()
