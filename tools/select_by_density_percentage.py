#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据密度值比例选择输入数据

该脚本根据给定的MRC文件和Coord.npy文件，计算每个坐标对应区域中
密度值高于contour阈值的比例，如果该比例低于指定百分比，则记录对应的区域编号。
结果将写入input_select.txt文件。
"""

import os
import argparse
import numpy as np
import mrcfile
from pathlib import Path
import matplotlib.pyplot as plt

def select_by_density_percentage(mrc_path, coord_path, contour, max_percentage, box_size=32, output_file="input_select.txt"):
    """
    根据密度值比例选择输入数据
    
    Parameters:
        mrc_path (str): MRC文件路径
        coord_path (str): Coord.npy文件路径
        contour (float): 用于分割背景和前景的阈值
        max_percentage (float): 最大允许的高于contour值的体素比例
        box_size (int): 切割的方块大小
        output_file (str): 输出文件路径
    """
    # 读取MRC文件
    print(f"读取MRC文件: {mrc_path}")
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        map_data = np.array(mrc.data, dtype=np.float32)
        print(f"MRC文件形状: {map_data.shape}")
        print(f"MRC数据范围: 最小值={np.min(map_data):.6f}, 最大值={np.max(map_data):.6f}, 均值={np.mean(map_data):.6f}")
    
    # 读取坐标文件
    print(f"读取坐标文件: {coord_path}")
    coords = np.load(coord_path)
    print(f"读取到{len(coords)}个坐标点")
    
    # 获取蛋白质名称
    protein_name = Path(mrc_path).stem
    
    # 记录符合条件的区域编号
    selected_indices = []
    
    # 统计结果
    total_boxes = len(coords)
    selected_boxes = 0
    
    # 处理进度显示
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    print(f"设定的contour阈值: {contour:.6f}")
    print(f"最大允许的高于contour值的体素比例: {max_percentage}%")
    print(f"处理坐标点...")
    
    # 存储每个区域的密度分布情况
    density_percentages = []
    
    # 遍历所有坐标
    iter_coords = tqdm(coords) if use_tqdm else coords
    for i, coord in enumerate(iter_coords):
        x_start, y_start, z_start = coord
        
        # 确定切割区域的结束坐标
        x_end = min(x_start + box_size, map_data.shape[0])
        y_end = min(y_start + box_size, map_data.shape[1])
        z_end = min(z_start + box_size, map_data.shape[2])
        
        # 检查坐标是否在MRC数据范围内
        if x_start >= map_data.shape[0] or y_start >= map_data.shape[1] or z_start >= map_data.shape[2]:
            print(f"警告: 坐标 {coord} 超出MRC数据范围 {map_data.shape}，跳过")
            continue
        
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
        
        # 记录所有区域的百分比情况
        density_percentages.append(percentage)
        
        # 检查该比例是否小于给定阈值
        if percentage < max_percentage:
            selected_indices.append((i, percentage))  # 记录索引和百分比
            selected_boxes += 1
    
    # 保存结果到文件
    with open(output_file, "w") as f:
        f.write(f"蛋白质: {protein_name}\n")
        f.write(f"Contour阈值: {contour:.6f}\n")
        f.write(f"最大允许区域中高于contour值的体素比例: {max_percentage}%\n")
        f.write(f"总坐标数: {total_boxes}\n")
        f.write(f"符合条件的坐标数: {selected_boxes}\n\n")
        f.write("解释: 以下区域中，高于contour值({contour:.6f})的体素比例都小于{max_percentage}%\n\n")
        f.write("符合条件的区域编号 - 实际高于contour的比例:\n")
        for idx, perc in selected_indices:
            coord = coords[idx]
            f.write(f"{idx} - 坐标: [{coord[0]}, {coord[1]}, {coord[2]}] - 高于contour比例: {perc:.2f}%\n")
    
    print(f"\n处理完成。")
    print(f"总坐标数: {total_boxes}")
    print(f"符合条件的坐标数: {selected_boxes}，占比: {selected_boxes/total_boxes*100:.2f}%")
    print(f"结果已保存到 {output_file}")
    
    # 可视化密度分布
    if len(density_percentages) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(density_percentages, bins=50, alpha=0.7)
        ax.axvline(x=max_percentage, color='r', linestyle='--', 
                label=f'阈值: {max_percentage:.2f}%')
        ax.set_title('区域密度分布直方图')
        ax.set_xlabel('高于contour值的体素百分比(%)')
        ax.set_ylabel('区域数量')
        ax.legend()
        
        # 保存图像
        plot_file = os.path.splitext(output_file)[0] + "_distribution.png"
        plt.savefig(plot_file)
        print(f"分布直方图已保存到 {plot_file}")
    
    return selected_indices

def main():
    parser = argparse.ArgumentParser(description="根据密度值比例选择输入数据")
    parser.add_argument("coord_path", help="Coord.npy文件路径")
    parser.add_argument("mrc_path", help="MRC文件路径")
    parser.add_argument("--contour", "-c", type=float, required=True,
                        help="用于分割背景和前景的阈值")
    parser.add_argument("--max_percentage", "-p", type=float, default=50.0, 
                        help="最大允许的高于contour值的体素比例，默认为50.0")
    parser.add_argument("--box_size", "-b", type=int, default=32,
                        help="切割的方块大小，默认为32")
    parser.add_argument("--output", "-o", default="input_select.txt",
                        help="输出文件路径，默认为input_select.txt")
    
    args = parser.parse_args()
    
    select_by_density_percentage(
        args.mrc_path, 
        args.coord_path, 
        args.contour,
        args.max_percentage,
        args.box_size,
        args.output
    )

if __name__ == "__main__":
    main()
