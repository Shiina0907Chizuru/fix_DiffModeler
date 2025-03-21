#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标准化MRC文件数据的脚本

该脚本读取MRC文件，并使用Z-score方法（减去均值，除以标准差）对数据进行标准化。
可以选择保存标准化后的MRC文件，并提供原始和标准化后数据的统计信息。
"""

import os
import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def normalize_map(map_data):
    """
    使用Z-score方法标准化地图：减去均值，除以标准差
    
    Parameters:
        map_data (numpy.ndarray): 输入地图数据

    Returns:
        numpy.ndarray: 标准化后的地图数据，均值为0，标准差为1
    """
    if map_data.std() != 0:
        # 标准Z-score归一化: 减去均值，除以标准差
        return (map_data - map_data.mean()) / map_data.std()
    else:
        # 如果标准差为0（所有值相同），则直接返回原始数据
        return map_data

def minmax_normalize(map_data, min_val=None, max_val=None):
    """
    使用最小-最大归一化方法标准化地图：将值映射到[0,1]范围
    
    Parameters:
        map_data (numpy.ndarray): 输入地图数据
        min_val (float, optional): 指定的最小值。如果为None，使用map_data的最小值
        max_val (float, optional): 指定的最大值。如果为None，使用map_data的最大值

    Returns:
        numpy.ndarray: 归一化后的地图数据，范围在[0,1]之间
    """
    if min_val is None:
        min_val = map_data.min()
    if max_val is None:
        max_val = map_data.max()
        
    if max_val != min_val:
        # 最小-最大归一化: (x - min) / (max - min)
        return (map_data - min_val) / (max_val - min_val)
    else:
        # 如果最大值等于最小值，返回全零数组
        return np.zeros_like(map_data)

def percentile_clip_normalize(map_data, lower_percentile=0, upper_percentile=98):
    """
    使用百分位截断后的最小-最大归一化方法
    
    Parameters:
        map_data (numpy.ndarray): 输入地图数据
        lower_percentile (float): 下截断百分位(0-100)
        upper_percentile (float): 上截断百分位(0-100)

    Returns:
        numpy.ndarray: 标准化后的地图数据，范围在[0,1]之间
    """
    # 计算上下百分位值
    lower_value = np.percentile(map_data, lower_percentile)
    upper_value = np.percentile(map_data, upper_percentile)
    
    print(f"下截断百分位({lower_percentile}%)值: {lower_value:.8f}")
    print(f"上截断百分位({upper_percentile}%)值: {upper_value:.8f}")
    
    # 截断数据
    clipped_data = map_data.copy()
    clipped_data[clipped_data < lower_value] = lower_value
    clipped_data[clipped_data > upper_value] = upper_value
    
    # 统计被截断的值
    num_lower_clipped = np.sum(map_data < lower_value)
    num_upper_clipped = np.sum(map_data > upper_value)
    total_pixels = map_data.size
    
    print(f"低于{lower_percentile}%百分位的值数量: {num_lower_clipped} ({num_lower_clipped/total_pixels*100:.4f}%)")
    print(f"高于{upper_percentile}%百分位的值数量: {num_upper_clipped} ({num_upper_clipped/total_pixels*100:.4f}%)")
    
    # 进行最小-最大归一化
    return minmax_normalize(clipped_data)

def process_mrc_file(input_path, output_path=None, method='zscore', lower_percentile=0, upper_percentile=98, 
                    plot=False, slice_idx=None):
    """
    处理MRC文件
    
    Parameters:
        input_path (str): 输入MRC文件路径
        output_path (str, optional): 输出MRC文件路径。如果为None，则不保存文件
        method (str): 标准化方法，可选'zscore', 'minmax', 'percentile'
        lower_percentile (float): 下截断百分位(0-100)，仅在method='percentile'时使用
        upper_percentile (float): 上截断百分位(0-100)，仅在method='percentile'时使用
        plot (bool): 是否绘制原始和标准化后的数据直方图
        slice_idx (int, optional): 要可视化的切片索引。如果为None，则使用中间切片
    """
    try:
        print(f"正在读取MRC文件: {input_path}")
        with mrcfile.open(input_path, permissive=True) as mrc:
            # 获取原始数据
            original_data = np.array(mrc.data, dtype=np.float32)
            
            # 如果原始数据是整型，可能需要转换
            if np.issubdtype(original_data.dtype, np.integer):
                original_data = original_data.astype(np.float32)
            
            # 获取原始数据的统计信息
            print("\n原始数据统计:")
            print(f"形状: {original_data.shape}")
            print(f"数据类型: {original_data.dtype}")
            print(f"最小值: {original_data.min():.8f}")
            print(f"最大值: {original_data.max():.8f}")
            print(f"均值: {original_data.mean():.8f}")
            print(f"标准差: {original_data.std():.8f}")
            print(f"非零元素数量: {np.count_nonzero(original_data)}")
            print(f"总元素数量: {original_data.size}")
            
            # 选择标准化方法
            if method == 'zscore':
                print("\n使用Z-score标准化方法...")
                standardized_data = normalize_map(original_data)
            elif method == 'minmax':
                print("\n使用最小-最大归一化方法...")
                standardized_data = minmax_normalize(original_data)
            elif method == 'percentile':
                print(f"\n使用百分位({lower_percentile}%-{upper_percentile}%)截断后的最小-最大归一化方法...")
                standardized_data = percentile_clip_normalize(original_data, lower_percentile, upper_percentile)
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            # 获取标准化后数据的统计信息
            print("\n标准化后数据统计:")
            print(f"最小值: {standardized_data.min():.8f}")
            print(f"最大值: {standardized_data.max():.8f}")
            print(f"均值: {standardized_data.mean():.8f}")
            print(f"标准差: {standardized_data.std():.8f}")
            
            # 如果指定了输出路径，保存标准化后的MRC文件
            if output_path:
                print(f"\n保存标准化后的MRC文件到: {output_path}")
                # 确保输出目录存在
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # 保存标准化后的数据到新的MRC文件
                with mrcfile.new(output_path, overwrite=True) as new_mrc:
                    # 设置新数据
                    new_mrc.set_data(standardized_data.astype(np.float32))
                    
                    # 复制部分原始MRC的头信息（避免尝试设置只读属性）
                    if hasattr(mrc, 'header'):
                        try:
                            # 尝试复制可写的header属性
                            if hasattr(mrc.header, 'mx'):
                                new_mrc.header.mx = mrc.header.mx
                            if hasattr(mrc.header, 'my'):
                                new_mrc.header.my = mrc.header.my
                            if hasattr(mrc.header, 'mz'):
                                new_mrc.header.mz = mrc.header.mz
                            if hasattr(mrc.header, 'cella'):
                                new_mrc.header.cella = mrc.header.cella
                            if hasattr(mrc.header, 'cellb'):
                                new_mrc.header.cellb = mrc.header.cellb
                            if hasattr(mrc.header, 'mapc'):
                                new_mrc.header.mapc = mrc.header.mapc
                            if hasattr(mrc.header, 'mapr'):
                                new_mrc.header.mapr = mrc.header.mapr
                            if hasattr(mrc.header, 'maps'):
                                new_mrc.header.maps = mrc.header.maps
                            
                            # 添加复制origin信息
                            if hasattr(mrc.header, 'origin'):
                                new_mrc.header.origin = mrc.header.origin
                            if hasattr(mrc.header, 'nxstart'):
                                new_mrc.header.nxstart = mrc.header.nxstart
                            if hasattr(mrc.header, 'nystart'):
                                new_mrc.header.nystart = mrc.header.nystart
                            if hasattr(mrc.header, 'nzstart'):
                                new_mrc.header.nzstart = mrc.header.nzstart
                            
                            # 打印origin信息以便验证
                            print("复制原始origin信息:")
                            if hasattr(mrc.header, 'origin'):
                                print(f"  原始origin: {mrc.header.origin}")
                            if hasattr(mrc.header, 'nxstart'):
                                print(f"  原始nxstart: {mrc.header.nxstart}")
                                print(f"  原始nystart: {mrc.header.nystart}")
                                print(f"  原始nzstart: {mrc.header.nzstart}")
                        except Exception as e:
                            print(f"  注意: 无法复制某些header属性: {str(e)}")
                    
                    # 更新统计信息
                    new_mrc.update_header_stats()
            
            # 绘制直方图和可视化切片
            if plot:
                # 选择要可视化的切片
                if slice_idx is None and len(original_data.shape) == 3:
                    slice_idx = original_data.shape[0] // 2  # 使用中间切片
                
                # 创建一个包含两行的图
                fig = plt.figure(figsize=(15, 12))
                
                # 绘制原始数据的直方图
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.hist(original_data.flatten(), bins=100, alpha=0.7)
                ax1.set_title('Original Data Histogram')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Frequency')
                
                # 绘制标准化后数据的直方图
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.hist(standardized_data.flatten(), bins=100, alpha=0.7)
                ax2.set_title(f'Standardized Data Histogram (Method: {method})')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Frequency')
                
                # 如果数据是3D的，显示一个切片
                if len(original_data.shape) == 3 and slice_idx is not None:
                    # 显示原始数据的切片
                    ax3 = fig.add_subplot(2, 2, 3)
                    im1 = ax3.imshow(original_data[slice_idx], cmap='viridis')
                    ax3.set_title(f'Original Data Slice (Index: {slice_idx})')
                    plt.colorbar(im1, ax=ax3)
                    
                    # 显示标准化后数据的切片
                    ax4 = fig.add_subplot(2, 2, 4)
                    im2 = ax4.imshow(standardized_data[slice_idx], cmap='viridis')
                    ax4.set_title(f'Standardized Data Slice (Index: {slice_idx})')
                    plt.colorbar(im2, ax=ax4)
                
                plt.tight_layout()
                
                # 保存图像
                plot_output = os.path.splitext(input_path)[0] + f"_{method}_normalized_plot.png"
                plt.savefig(plot_output)
                print(f"\n已保存可视化图像到: {plot_output}")
                
                # 显示图像
                plt.show()
                
    except Exception as e:
        print(f"处理MRC文件时出错: {str(e)}")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='标准化MRC文件的数据')
    parser.add_argument('input_path', type=str, help='输入MRC文件路径')
    parser.add_argument('--output_path', '-o', type=str, help='输出MRC文件路径（如果不指定，则自动生成）')
    parser.add_argument('--method', '-m', type=str, choices=['zscore', 'minmax', 'percentile'], default='zscore',
                        help='标准化方法: zscore (Z-score标准化), minmax (最小-最大归一化), percentile (百分位截断后归一化)')
    parser.add_argument('--lower_percentile', type=float, default=0,
                        help='下截断百分位 (0-100), 仅在method=percentile时使用')
    parser.add_argument('--upper_percentile', type=float, default=98,
                        help='上截断百分位 (0-100), 仅在method=percentile时使用')
    parser.add_argument('--plot', '-p', action='store_true', 
                        help='是否绘制原始和标准化后的数据直方图和切片可视化')
    parser.add_argument('--slice_idx', '-s', type=int, 
                        help='要可视化的切片索引。如果不指定，则使用中间切片')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，则自动生成一个
    if args.output_path is None:
        input_base = os.path.splitext(args.input_path)[0]
        args.output_path = f"{input_base}_{args.method}_normalized.mrc"
    
    # 处理MRC文件
    process_mrc_file(
        args.input_path, 
        args.output_path, 
        args.method, 
        args.lower_percentile, 
        args.upper_percentile, 
        args.plot, 
        args.slice_idx
    )
    
    print(f"\n脚本执行完成！")
    print(f"使用了 {args.method} 方法标准化 MRC 文件。")
    print(f"标准化后的 MRC 文件已保存到: {os.path.abspath(args.output_path)}")
    
if __name__ == "__main__":
    main()
