#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从大阈值文件中提取指定蛋白质的阈值信息")
    parser.add_argument("--threshold_file", required=True, help="包含所有阈值信息的大文件")
    parser.add_argument("--info_file", required=True, help="包含需要提取的蛋白质ID的信息文件")
    parser.add_argument("--output_file", required=True, help="输出的小阈值信息文件路径")
    parser.add_argument("--include_all_values", action="store_true", 
                        help="是否包含所有值（包括可能的CA原子数量等）")
    return parser.parse_args()

def read_protein_ids(file_path):
    """
    从信息文件中读取蛋白质ID列表
    
    参数:
    - file_path: 信息文件路径
    
    返回:
    - protein_ids: 蛋白质ID集合
    """
    protein_ids = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 提取ID部分
                parts = line.split(':', 1)
                if len(parts) >= 1:
                    protein_id = parts[0].strip().lower()
                    protein_ids.add(protein_id)
    except Exception as e:
        print(f"读取信息文件时出错: {str(e)}")
    
    return protein_ids

def extract_thresholds(threshold_file, info_file, output_file, include_all_values=False):
    """
    从大阈值文件中提取特定蛋白质的阈值信息
    
    参数:
    - threshold_file: 包含所有阈值信息的大文件
    - info_file: 包含需要提取的蛋白质ID的信息文件
    - output_file: 输出的小阈值信息文件路径
    - include_all_values: 是否包含所有值（包括可能的CA原子数量等）
    """
    # 读取需要提取的蛋白质ID
    target_ids = read_protein_ids(info_file)
    print(f"从信息文件中读取了{len(target_ids)}个蛋白质ID")
    
    # 从大阈值文件中提取指定蛋白质的阈值信息
    extracted_entries = []
    
    try:
        with open(threshold_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 处理注释行
                if line.startswith('#'):
                    extracted_entries.append(line)
                    continue
                
                # 解析行内容
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    protein_id = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    # 如果是目标蛋白质，或者需要包含所有值（对于值 > 1 的条目，可能是原子数量）
                    if protein_id in target_ids:
                        extracted_entries.append(line)
                    elif include_all_values and float(value) > 1.0:
                        # 假设大于1的值可能是CA原子数量
                        if protein_id in target_ids:
                            extracted_entries.append(line)
    except Exception as e:
        print(f"处理阈值文件时出错: {str(e)}")
    
    # 写入提取的阈值信息到输出文件
    try:
        with open(output_file, 'w') as f:
            # 添加标题注释
            if not any(line.startswith('#') for line in extracted_entries):
                f.write("# 蛋白质ID: 阈值\n")
            
            # 写入提取的条目
            for entry in extracted_entries:
                f.write(f"{entry}\n")
        
        print(f"已提取{len(extracted_entries) - (1 if extracted_entries and extracted_entries[0].startswith('#') else 0)}个阈值信息")
        print(f"结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"写入输出文件时出错: {str(e)}")

def main():
    args = parse_args()
    extract_thresholds(args.threshold_file, args.info_file, args.output_file, args.include_all_values)

if __name__ == "__main__":
    main()
