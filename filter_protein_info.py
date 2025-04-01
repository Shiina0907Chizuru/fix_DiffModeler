#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
from datetime import datetime

def parse_protein_names(file_path):
    """
    从文件中解析蛋白质名称
    
    参数:
    - file_path: 文件路径
    
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
                print(f"成功使用 {encoding} 编码读取文件: {os.path.basename(file_path)}")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码读取失败，尝试下一种编码")
                continue
            except Exception as e:
                print(f"读取文件时出错: {str(e)}")
                return set()
        
        if file_content is None:
            print(f"无法读取文件: {file_path}")
            return set()
        
        for line in file_content:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 尝试多种模式匹配蛋白质名称
            # 模式1: 6adq: 0.6389
            match1 = re.match(r'^(\w+):?\s+[\d\.]+', line)
            # 模式2: 只有蛋白质名称
            match2 = re.match(r'^(\w+)$', line)
            
            if match1:
                protein_name = match1.group(1).strip()
                proteins.add(protein_name)
            elif match2:
                protein_name = match2.group(1).strip()
                proteins.add(protein_name)
            else:
                # 尝试直接分割并获取第一部分
                parts = line.split()
                if len(parts) >= 1:
                    protein_name = parts[0].strip().rstrip(':')
                    if re.match(r'^\w+$', protein_name):  # 确保是有效的蛋白质名称
                        proteins.add(protein_name)
    
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    return proteins

def filter_protein_info(large_file, small_file, output_file):
    """
    过滤蛋白质信息，生成新的信息文件
    
    参数:
    - large_file: 大信息文件路径
    - small_file: 小信息文件路径，包含要跳过的蛋白质
    - output_file: 输出文件路径
    """
    # 解析小信息文件中的蛋白质名称
    skip_proteins = parse_protein_names(small_file)
    print(f"从小信息文件中解析到 {len(skip_proteins)} 个要跳过的蛋白质")
    
    # 解析大信息文件中的蛋白质名称
    all_proteins = parse_protein_names(large_file)
    print(f"从大信息文件中解析到 {len(all_proteins)} 个蛋白质")
    
    # 过滤掉小信息文件中的蛋白质
    filtered_proteins = all_proteins - skip_proteins
    print(f"过滤后剩余 {len(filtered_proteins)} 个蛋白质")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果没有指定输出文件，则生成默认名称
    if not output_file:
        output_dir = os.path.dirname(large_file)
        output_file = os.path.join(output_dir, f"filtered_proteins_{timestamp}.txt")
    
    # 将过滤后的蛋白质名称写入新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 不写入注释和空行，直接写入蛋白质名称
            # 按字母顺序排序
            for protein in sorted(filtered_proteins):
                f.write(f"{protein}\n")
        
        print(f"已将 {len(filtered_proteins)} 个过滤后的蛋白质名称保存至: {output_file}")
    except Exception as e:
        print(f"保存过滤后的蛋白质列表时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='过滤蛋白质信息文件，生成新的信息文件')
    parser.add_argument('--large_file', type=str, required=True, help='大信息文件路径，包含所有蛋白质')
    parser.add_argument('--small_file', type=str, required=True, help='小信息文件路径，包含要跳过的蛋白质')
    parser.add_argument('--output', type=str, help='输出文件路径，默认为filtered_proteins_时间戳.txt')
    
    args = parser.parse_args()
    
    print(f"开始处理:")
    print(f"大信息文件: {args.large_file}")
    print(f"小信息文件: {args.small_file}")
    
    # 过滤蛋白质信息
    filter_protein_info(args.large_file, args.small_file, args.output)
    
    print("处理完成")

if __name__ == "__main__":
    main()
