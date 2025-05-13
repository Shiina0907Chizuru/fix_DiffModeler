#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import glob
import re
try:
    import mrcfile
except ImportError:
    print("警告: 未找到mrcfile库，请安装: pip install mrcfile")
    print("正在尝试继续执行...")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="计算MRC文件的3 sigma contour值")
    parser.add_argument("--input_dir", required=True, help="包含.mrc文件的目录路径")
    parser.add_argument("--output_file", default="mrc_contour_values.txt", help="输出文件路径，默认为mrc_contour_values.txt")
    parser.add_argument("--pattern", default="*.mrc", help="文件匹配模式，默认为*.mrc")
    parser.add_argument("--recursive", action="store_true", help="是否递归搜索子目录")
    parser.add_argument("--sigma", type=float, default=3.0, help="使用几倍标准差作为contour值，默认为3.0")
    return parser.parse_args()

def extract_pdb_id(filename):
    """
    从文件名中提取PDB ID
    例如: 7jlu_backbone.mrc -> 7jlu
    """
    # 尝试使用常见的模式提取PDB ID
    # 方法1: 提取文件名开头的字母数字组合(通常是PDB ID)
    base_name = os.path.basename(filename)
    match = re.match(r'^(\w+)[-_]', base_name)
    if match:
        return match.group(1).lower()
    
    # 方法2: 如果文件名不包含下划线，直接取文件名(不含扩展名)
    name_without_ext = os.path.splitext(base_name)[0]
    if '_' not in name_without_ext and '-' not in name_without_ext:
        return name_without_ext.lower()
    
    # 方法3: 使用目录名提取
    parent_dir = os.path.basename(os.path.dirname(filename))
    match = re.search(r'PDB-(\w+)-', parent_dir)
    if match:
        return match.group(1).lower()
    
    # 默认返回文件名(不含扩展名)
    return name_without_ext.lower()

def calculate_contour(mrc_file, sigma_factor=3.0):
    """
    计算MRC文件的contour值(3倍标准差)
    
    参数:
    - mrc_file: MRC文件路径
    - sigma_factor: 使用几倍标准差，默认为3.0
    
    返回:
    - contour: 计算得到的contour值
    """
    try:
        with mrcfile.open(mrc_file) as mrc:
            # 计算数据的标准差
            data = mrc.data
            sigma = np.std(data)
            # 计算contour值 (3 sigma)
            contour = sigma_factor * sigma
            return contour
    except Exception as e:
        print(f"处理文件 {mrc_file} 时出错: {str(e)}")
        return None

def find_mrc_files(input_dir, pattern="*.mrc", recursive=False):
    """
    在目录中查找MRC文件
    
    参数:
    - input_dir: 输入目录
    - pattern: 文件匹配模式
    - recursive: 是否递归搜索子目录
    
    返回:
    - mrc_files: 找到的MRC文件列表
    """
    if recursive:
        # 递归搜索
        search_pattern = os.path.join(input_dir, "**", pattern)
        mrc_files = glob.glob(search_pattern, recursive=True)
    else:
        # 只在当前目录搜索
        search_pattern = os.path.join(input_dir, pattern)
        mrc_files = glob.glob(search_pattern)
    
    return mrc_files

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 目录 {args.input_dir} 不存在")
        return
    
    # 查找MRC文件
    print(f"正在目录 {args.input_dir} 中查找MRC文件...")
    mrc_files = find_mrc_files(args.input_dir, args.pattern, args.recursive)
    
    if not mrc_files:
        print(f"错误: 在目录 {args.input_dir} 中未找到任何MRC文件")
        return
    
    print(f"找到 {len(mrc_files)} 个MRC文件")
    
    # 计算每个文件的contour值
    contour_values = {}
    for mrc_file in mrc_files:
        pdb_id = extract_pdb_id(mrc_file)
        contour = calculate_contour(mrc_file, args.sigma)
        
        if contour is not None:
            contour_values[pdb_id] = contour
            print(f"文件: {os.path.basename(mrc_file)}, PDB ID: {pdb_id}, Contour ({args.sigma} sigma): {contour:.6f}")
    
    # 将结果写入输出文件
    with open(args.output_file, 'w') as f:
        for pdb_id in sorted(contour_values.keys()):
            f.write(f"{pdb_id}: {contour_values[pdb_id]:.6f}\n")
    
    print(f"已成功计算 {len(contour_values)}/{len(mrc_files)} 个MRC文件的contour值，并保存到 {args.output_file}")

if __name__ == "__main__":
    main()
