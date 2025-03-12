#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
裁剪MRC文件数据到指定百分位

该脚本读取MRC文件，计算指定百分位（默认98%）的值，
并将所有大于该百分位的值设置为该百分位值，然后保存为新的MRC文件。
"""

import os
import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def clip_map_to_percentile(map_data, percentile=98):
    """
    将地图数据裁剪到指定百分位
    
    Parameters:
        map_data (numpy.ndarray): 输入地图数据
        percentile (float): 百分位值 (0-100)

    Returns:
        numpy.ndarray: 裁剪后的地图数据
        float: 计算的百分位值
    """
    # 计算百分位值
    percentile_value = np.percentile(map_data, percentile)
    print(f"map hist log percentage {percentile}: {percentile_value}")
    
    # 创建处理后的数据副本
    clipped_data = map_data.copy()
    
    # 将大于百分位值的数据设置为百分位值
    mask = clipped_data > percentile_value
    clipped_data[mask] = percentile_value
    
    # 统计修改的数据点数量和百分比
    num_modified = np.sum(mask)
    percent_modified = (num_modified / map_data.size) * 100
    print(f"修改的数据点数量: {num_modified} ({percent_modified:.4f}%)")
    
    return clipped_data, percentile_value

def process_mrc_file(input_path, output_path=None, percentile=98, plot=False, slice_idx=None):
    """
    处理MRC文件
    
    Parameters:
        input_path (str): 输入MRC文件路径
        output_path (str, optional): 输出MRC文件路径。如果为None，则不保存文件
        percentile (float): 百分位值 (0-100)
        plot (bool): 是否绘制原始和处理后的数据直方图和切片
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
            
            # 计算并打印更多百分位值
            percentiles = [90, 95, 98, 99, 99.5, 99.9]
            for p in percentiles:
                p_value = np.percentile(original_data, p)
                print(f"{p}%百分位值: {p_value:.8f}")
            
            # 将数据裁剪到指定百分位
            print(f"\n裁剪数据到{percentile}%百分位...")
            clipped_data, percentile_value = clip_map_to_percentile(original_data, percentile)
            
            # 获取处理后数据的统计信息
            print("\n处理后数据统计:")
            print(f"最小值: {clipped_data.min():.8f}")
            print(f"最大值: {clipped_data.max():.8f}")
            print(f"均值: {clipped_data.mean():.8f}")
            print(f"标准差: {clipped_data.std():.8f}")
            
            # 如果指定了输出路径，保存处理后的MRC文件
            if output_path:
                print(f"\n保存处理后的MRC文件到: {output_path}")
                # 确保输出目录存在
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # 保存处理后的数据到新的MRC文件
                with mrcfile.new(output_path, overwrite=True) as new_mrc:
                    # 设置新数据
                    new_mrc.set_data(clipped_data.astype(np.float32))
                    
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
                        except Exception as e:
                            print(f"  注意: 无法复制某些header属性: {str(e)}")
                    
                    # 更新统计信息
                    new_mrc.update_header_stats()
            
            # 绘制直方图和可视化切片
            if plot:
                # 选择要可视化的切片
                if slice_idx is None and len(original_data.shape) == 3:
                    slice_idx = original_data.shape[0] // 2  # 使用中间切片
                
                # 创建一个包含两行两列的图
                fig = plt.figure(figsize=(15, 12))
                
                # 绘制原始数据的直方图
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.hist(original_data.flatten(), bins=100, alpha=0.7)
                ax1.axvline(x=percentile_value, color='r', linestyle='--', 
                           label=f'{percentile}% percentile: {percentile_value:.3f}')
                ax1.set_title('Original Data Histogram')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                
                # 绘制处理后数据的直方图
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.hist(clipped_data.flatten(), bins=100, alpha=0.7)
                ax2.axvline(x=percentile_value, color='r', linestyle='--', 
                           label=f'{percentile}% percentile: {percentile_value:.3f}')
                ax2.set_title('Clipped Data Histogram')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                
                # 如果数据是3D的，显示切片
                if len(original_data.shape) == 3 and slice_idx is not None:
                    # 显示原始数据切片
                    ax3 = fig.add_subplot(2, 2, 3)
                    im3 = ax3.imshow(original_data[slice_idx], cmap='viridis')
                    ax3.set_title(f'Original Data (Slice {slice_idx})')
                    plt.colorbar(im3, ax=ax3)
                    
                    # 显示处理后数据切片
                    ax4 = fig.add_subplot(2, 2, 4)
                    im4 = ax4.imshow(clipped_data[slice_idx], cmap='viridis')
                    ax4.set_title(f'Clipped Data (Slice {slice_idx})')
                    plt.colorbar(im4, ax=ax4)
                
                plt.tight_layout()
                
                # 如果指定了输出路径，保存图像
                if output_path:
                    plot_path = os.path.splitext(output_path)[0] + "_plot.png"
                    plt.savefig(plot_path, dpi=300)
                    print(f"保存图像到: {plot_path}")
                
                plt.show()
            
            return percentile_value, clipped_data
            
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description="裁剪MRC文件数据到指定百分位")
    parser.add_argument("input_path", help="输入MRC文件路径")
    parser.add_argument("--output_path", "-o", help="输出MRC文件路径")
    parser.add_argument("--percentile", "-p", type=float, default=98, 
                        help="百分位值 (0-100)，默认为98")
    parser.add_argument("--plot", action="store_true", help="是否绘制直方图和切片")
    parser.add_argument("--slice_idx", type=int, help="要可视化的切片索引")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，使用输入路径添加后缀
    if not args.output_path and args.input_path:
        file_name, file_ext = os.path.splitext(args.input_path)
        args.output_path = f"{file_name}_p{args.percentile}{file_ext}"
    
    # 处理MRC文件
    percentile_value, _ = process_mrc_file(
        args.input_path, args.output_path, args.percentile, args.plot, args.slice_idx
    )
    
    if percentile_value is not None:
        print(f"\n处理完成。{args.percentile}%百分位值: {percentile_value:.8f}")
        if args.output_path:
            print(f"处理后的文件已保存到: {args.output_path}")
    else:
        print("处理失败。")

if __name__ == "__main__":
    main()
