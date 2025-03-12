#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量处理多个蛋白质MRC文件

该脚本读取一个包含蛋白质名称和对应contour值的信息文件，
然后批量处理指定目录下的所有蛋白质MRC文件，分析每个区域中
高于contour值的体素比例，并生成满足条件的区域编号列表。

注意：此脚本默认使用box_size=64来处理数据块，与generate_dataset.py中的默认参数保持一致。
"""

import os
import sys
import argparse
import numpy as np
import mrcfile
from pathlib import Path
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


def parse_protein_info(info_file):
    """
    解析蛋白质信息文件
    
    Parameters:
        info_file (str): 包含蛋白质名称和contour值的文件路径
        
    Returns:
        dict: 蛋白质名称到contour值的映射
    """
    protein_contours = {}
    
    with open(info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 支持冒号分隔格式 (如 "2ygd: 0.004")
            if ':' in line:
                parts = line.split(':', 1)
                protein_name = parts[0].strip()
                try:
                    contour = float(parts[1].strip())
                    protein_contours[protein_name] = contour
                except ValueError:
                    print(f"警告: 无法解析contour值: {parts[1]}，跳过蛋白质 {protein_name}")
            # 支持空格分隔格式 (如 "2ygd 0.004")
            else:
                parts = line.split()
                if len(parts) >= 2:
                    protein_name = parts[0]
                    try:
                        contour = float(parts[1])
                        protein_contours[protein_name] = contour
                    except ValueError:
                        print(f"警告: 无法解析contour值: {parts[1]}，跳过蛋白质 {protein_name}")
    
    return protein_contours


def find_mrc_and_coord(protein_folder, protein_contours):
    """
    查找蛋白质文件夹中的MRC和坐标文件
    
    Parameters:
        protein_folder (str): 蛋白质文件夹路径
        protein_contours (dict): 蛋白质ID到contour值的映射字典
        
    Returns:
        tuple: (mrc_path, coord_path, pdb_id, emd_id, contour)
    """
    # 获取PDB ID和EMD ID
    folder_name = os.path.basename(protein_folder)
    match = re.match(r'PDB-([a-z0-9]+)-EMD-(\d+)', folder_name)
    
    if not match:
        print(f"警告: 无法从文件夹名称 {folder_name} 中提取PDB和EMD ID")
        return None, None, None, None, None
    
    pdb_id, emd_id = match.groups()
    
    # 查找MRC文件
    processed_dir = os.path.join(protein_folder, "processed")
    if not os.path.exists(processed_dir) or not os.path.isdir(processed_dir):
        # 如果processed目录不存在，尝试在根目录中查找
        mrc_files = glob.glob(os.path.join(protein_folder, f"{pdb_id}.mrc"))
        if not mrc_files:
            print(f"警告: 找不到蛋白质 {pdb_id} 的MRC文件")
            return None, None, pdb_id, emd_id, None
        mrc_path = mrc_files[0]
    else:
        # 在processed目录中查找
        mrc_files = glob.glob(os.path.join(processed_dir, f"{pdb_id}.mrc"))
        if not mrc_files:
            # 尝试查找增强后的MRC文件
            mrc_files = glob.glob(os.path.join(processed_dir, f"{pdb_id}_unified.mrc"))
        if not mrc_files:
            print(f"警告: 找不到蛋白质 {pdb_id} 的MRC文件")
            return None, None, pdb_id, emd_id, None
        mrc_path = mrc_files[0]
    
    # 仅在backbone_Dataset目录中查找Coord.npy文件
    backbone_dataset_dir = os.path.join(protein_folder, "backbone_Dataset")
    if not os.path.exists(backbone_dataset_dir) or not os.path.isdir(backbone_dataset_dir):
        print(f"警告: 找不到蛋白质 {pdb_id} 的backbone_Dataset目录")
        return None, None, pdb_id, emd_id, None
    
    coord_path = os.path.join(backbone_dataset_dir, "Coord.npy")
    if not os.path.exists(coord_path):
        print(f"警告: 找不到蛋白质 {pdb_id} 的Coord.npy文件: {coord_path}")
        return None, None, pdb_id, emd_id, None
    
    # 获取contour值
    contour = protein_contours.get(pdb_id)
    if contour is None:
        print(f"警告: 找不到蛋白质 {pdb_id} 的contour值")
        return None, None, pdb_id, emd_id, None
    
    print(f"蛋白质 {pdb_id} 使用contour值 {contour} (来自 {pdb_id})")
    
    return mrc_path, coord_path, pdb_id, emd_id, contour


def select_by_density(mrc_path, coord_path, contour, max_percentage, box_size=64):
    """
    根据密度值比例选择区域
    
    Parameters:
        mrc_path (str): MRC文件路径
        coord_path (str): Coord.npy文件路径
        contour (float): 用于分割背景和前景的阈值
        max_percentage (float): 最大允许的高于contour值的体素比例
        box_size (int): 切割的方块大小（默认为64，与generate_dataset.py中的默认参数保持一致）
        
    Returns:
        tuple: (selected_indices, density_percentages)
            selected_indices: 满足条件的区域索引和百分比的列表 [(idx, percentage), ...]
            density_percentages: 所有区域的密度百分比列表
    """
    # 读取MRC文件
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            map_data = np.array(mrc.data, dtype=np.float32)
    except Exception as e:
        print(f"错误: 无法读取MRC文件 {mrc_path}: {str(e)}")
        return [], []
    
    # 读取坐标文件
    try:
        coords = np.load(coord_path)
    except Exception as e:
        print(f"错误: 无法读取Coord.npy文件 {coord_path}: {str(e)}")
        return [], []
    
    # 记录符合条件的区域编号及其密度分布
    selected_indices = []
    density_percentages = []
    
    # 遍历所有坐标
    for i, coord in enumerate(coords):
        x_start, y_start, z_start = coord
        
        # 检查坐标是否在MRC数据范围内
        if (x_start >= map_data.shape[0] or y_start >= map_data.shape[1] or 
            z_start >= map_data.shape[2]):
            continue
        
        # 确定切割区域的结束坐标
        x_end = min(x_start + box_size, map_data.shape[0])
        y_end = min(y_start + box_size, map_data.shape[1])
        z_end = min(z_start + box_size, map_data.shape[2])
        
        # 获取该区域数据
        segment = map_data[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # 确保区域大小与box_size一致
        if segment.shape != (box_size, box_size, box_size):
            padded_segment = np.zeros((box_size, box_size, box_size), dtype=segment.dtype)
            padded_segment[:segment.shape[0], :segment.shape[1], :segment.shape[2]] = segment
            segment = padded_segment
        
        # 计算高于contour值的体素比例
        above_contour = np.sum(segment > contour)
        total_voxels = box_size**3
        percentage = (above_contour / total_voxels) * 100
        
        # 记录密度百分比
        density_percentages.append(percentage)
        
        # 检查该比例是否小于给定阈值
        if percentage < max_percentage:
            selected_indices.append((i, percentage))
    
    return selected_indices, density_percentages


def generate_protein_report(protein_folder, selected_indices, density_percentages, 
                           contour, max_percentage, pdb_id, emd_id):
    """
    生成单个蛋白质的报告文件
    
    Parameters:
        protein_folder (str): 蛋白质文件夹路径
        selected_indices (list): 满足条件的区域索引和百分比 [(idx, percentage), ...]
        density_percentages (list): 所有区域的密度百分比
        contour (float): 使用的contour值
        max_percentage (float): 使用的最大百分比阈值
        pdb_id (str): PDB ID
        emd_id (str): EMD ID
        
    Returns:
        str: 报告文件路径
    """
    # 准备输出文件路径
    output_file = os.path.join(protein_folder, f"{pdb_id}_density_select.txt")
    
    # 保存结果到文件
    with open(output_file, "w") as f:
        f.write(f"蛋白质: {pdb_id} (EMD-{emd_id})\n")
        f.write(f"Contour阈值: {contour:.6f}\n")
        f.write(f"最大允许区域中高于contour值的体素比例: {max_percentage}%\n")
        f.write(f"总区域数: {len(density_percentages)}\n")
        f.write(f"符合条件的区域数: {len(selected_indices)}\n\n")
        f.write("解释: 以下区域中，高于contour值的体素比例都小于{0}%\n\n".format(max_percentage))
        f.write("符合条件的区域编号 - 实际高于contour的比例:\n")
        
        for idx, perc in selected_indices:
            f.write(f"{idx} - 高于contour比例: {perc:.2f}%\n")
    
    # 绘制密度分布图
    if density_percentages:
        plt.figure(figsize=(10, 6))
        plt.hist(density_percentages, bins=50, alpha=0.7)
        plt.axvline(x=max_percentage, color='r', linestyle='--', 
                   label=f'阈值: {max_percentage:.2f}%')
        plt.title(f'区域密度分布直方图 - {pdb_id}')
        plt.xlabel('高于contour值的体素百分比(%)')
        plt.ylabel('区域数量')
        plt.legend()
        
        # 保存图像
        plot_file = os.path.join(protein_folder, f"{pdb_id}_density_distribution.png")
        plt.savefig(plot_file)
        plt.close()
    
    return output_file


def generate_summary_report(base_dir, protein_results):
    """
    生成汇总报告
    
    Parameters:
        base_dir (str): 基础目录
        protein_results (dict): 蛋白质结果数据 {protein_name: [(idx, percentage), ...], ...}
        
    Returns:
        str: 汇总报告文件路径
    """
    summary_file = os.path.join(base_dir, "all_proteins_density_select.txt")
    
    with open(summary_file, "w") as f:
        f.write(f"密度选择汇总报告\n")
        f.write(f"处理时间: {os.path.basename(os.path.dirname(base_dir))}\n\n")
        f.write(f"总蛋白质数: {len(protein_results)}\n\n")
        
        for protein, results in protein_results.items():
            f.write(f"蛋白质: {protein}\n")
            if results:
                # 只写入索引，不写入百分比
                indices = [str(idx) for idx, _ in results]
                f.write(f"{protein}: {', '.join(indices)}\n")
            else:
                f.write(f"{protein}: 没有符合条件的区域\n")
    
    return summary_file


def process_dataset(dataset_dir, info_file, max_percentage=50.0, box_size=64):
    """
    处理整个数据集
    
    Parameters:
        dataset_dir (str): 数据集目录
        info_file (str): 包含蛋白质名称和contour值的文件路径
        max_percentage (float): 最大允许的高于contour值的体素比例
        box_size (int): 切割的方块大小（默认为64，与generate_dataset.py中的默认参数保持一致）
        
    Returns:
        str: 汇总报告文件路径
    """
    # 解析蛋白质信息文件
    protein_contours = parse_protein_info(info_file)
    if not protein_contours:
        print(f"错误: 没有从 {info_file} 提取到有效的蛋白质信息")
        return None
    
    print(f"从信息文件中读取了 {len(protein_contours)} 个蛋白质的contour值")
    
    # 检查数据集目录
    if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
        print(f"错误: 数据集目录不存在或不是目录: {dataset_dir}")
        return None
    
    # 查找origin目录
    origin_dir = os.path.join(dataset_dir, "origin")
    if not os.path.exists(origin_dir) or not os.path.isdir(origin_dir):
        print(f"警告: origin目录不存在，使用数据集目录: {dataset_dir}")
        origin_dir = dataset_dir
    
    # 查找所有蛋白质文件夹
    protein_folders = []
    for item in os.listdir(origin_dir):
        item_path = os.path.join(origin_dir, item)
        if os.path.isdir(item_path) and re.match(r'PDB-[a-z0-9]+-EMD-\d+', item):
            protein_folders.append(item_path)
    
    if not protein_folders:
        print(f"错误: 未找到符合格式的蛋白质文件夹")
        return None
    
    print(f"找到 {len(protein_folders)} 个蛋白质文件夹")
    
    # 存储所有蛋白质的结果
    protein_results = {}
    success_count = 0
    error_count = 0
    
    # 创建详细错误日志文件
    error_log_path = os.path.join(dataset_dir, "density_select_errors.log")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"密度选择处理错误日志 - {os.path.basename(dataset_dir)}\n")
        error_log.write(f"处理时间: {os.path.basename(os.path.dirname(dataset_dir))}\n\n")
    
    # 处理每个蛋白质文件夹
    for folder in tqdm(protein_folders, desc="处理蛋白质"):
        folder_name = os.path.basename(folder)
        
        # 查找MRC、坐标文件和对应的contour值
        mrc_path, coord_path, pdb_id, emd_id, contour = find_mrc_and_coord(folder, protein_contours)
        if not mrc_path or not coord_path or not contour:
            print(f"警告: 找不到蛋白质 {folder_name} 的必要文件或contour值，跳过")
            continue
        
        # 分析密度
        selected_indices, density_percentages = select_by_density(
            mrc_path, coord_path, contour, max_percentage, box_size
        )
        
        # 生成蛋白质报告
        report_file = generate_protein_report(
            folder, selected_indices, density_percentages, 
            contour, max_percentage, pdb_id, emd_id
        )
        
        print(f"蛋白质 {pdb_id} 处理完成，有 {len(selected_indices)}/{len(density_percentages)} 个区域满足条件")
        
        # 保存结果
        protein_results[pdb_id] = selected_indices
    
    # 生成汇总报告
    if protein_results:
        summary_file = generate_summary_report(dataset_dir, protein_results)
        print(f"汇总报告已保存到: {summary_file}")
        return summary_file
    else:
        print("错误: 没有任何蛋白质处理成功")
        return None


def main():
    parser = argparse.ArgumentParser(description="批量处理多个蛋白质MRC文件")
    parser.add_argument("dataset_dir", help="数据集目录")
    parser.add_argument("info_file", help="包含蛋白质名称和contour值的文件路径")
    parser.add_argument("--max_percentage", "-p", type=float, default=50.0, 
                        help="最大允许的高于contour值的体素比例，默认为50.0")
    parser.add_argument("--box_size", "-b", type=int, default=64,
                        help="切割的方块大小，默认为64，与generate_dataset.py中的默认参数保持一致")
    
    args = parser.parse_args()
    
    summary_file = process_dataset(
        args.dataset_dir,
        args.info_file,
        args.max_percentage,
        args.box_size
    )
    
    if summary_file:
        print(f"处理完成。汇总报告: {summary_file}")
    else:
        print("处理失败。")
        sys.exit(1)


if __name__ == "__main__":
    main()
# python batch_select_by_density.py /defaultShare/zcan-library/Diffmodeler_data/20250306dataset /path/to/your/contour_info_file.txt --max_percentage 50.0