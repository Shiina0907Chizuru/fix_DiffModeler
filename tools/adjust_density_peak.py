#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调整MRC文件的密度分布

该脚本将segment MRC文件中的密度峰值及其以下的所有值设为最小值，
使数据分布更加离散，背景与前景更加分明。
"""

import os
import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from scipy import stats

def find_density_peak(data, bins=1000, smooth=True, window_size=5):
    """
    找到密度分布的峰值
    
    Parameters:
        data (numpy.ndarray): 输入数据
        bins (int): 直方图的bin数量
        smooth (bool): 是否对直方图进行平滑处理
        window_size (int): 平滑窗口大小
        
    Returns:
        float: 密度峰值对应的数据值
    """
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 如果需要平滑
    if smooth and window_size > 1:
        # 使用简单的滑动平均进行平滑
        kernel = np.ones(window_size) / window_size
        hist_smooth = np.convolve(hist, kernel, mode='same')
    else:
        hist_smooth = hist
    
    # 找到峰值
    peak_idx = np.argmax(hist_smooth)
    peak_value = bin_centers[peak_idx]
    
    print(f"找到密度峰值: {peak_value:.6f} (索引: {peak_idx}, 计数: {hist[peak_idx]})")
    
    return peak_value

def process_mrc_file(input_path, output_path=None, threshold=None, set_to_min=True, plot=False):
    """
    处理MRC文件，将峰值及其以下的值设为最小值
    
    Parameters:
        input_path (str): 输入MRC文件路径
        output_path (str): 输出MRC文件路径
        threshold (float): 阈值，如果不指定则自动找到密度峰值
        set_to_min (bool): 如果为True，则设置为最小值，否则设为0
        plot (bool): 是否绘制处理前后的直方图
    """
    try:
        print(f"正在读取MRC文件: {input_path}")
        with mrcfile.open(input_path, permissive=True) as mrc:
            # 获取原始数据
            original_data = np.array(mrc.data, dtype=np.float32)
            
            # 获取原始数据的统计信息
            print("\n原始数据统计:")
            print(f"形状: {original_data.shape}")
            print(f"数据类型: {original_data.dtype}")
            print(f"最小值: {original_data.min():.8f}")
            print(f"最大值: {original_data.max():.8f}")
            print(f"均值: {original_data.mean():.8f}")
            print(f"标准差: {original_data.std():.8f}")
            
            # 如果没有指定阈值，找到密度峰值
            if threshold is None:
                threshold = find_density_peak(original_data)
            
            # 创建处理后的数据副本
            processed_data = original_data.copy()
            
            # 将小于等于阈值的所有点设为最小值或0
            if set_to_min:
                min_value = original_data.min()
                mask = processed_data <= threshold
                processed_data[mask] = min_value
            else:
                mask = processed_data <= threshold
                processed_data[mask] = 0
            
            # 统计修改的数据点数量和百分比
            num_modified = np.sum(mask)
            percent_modified = (num_modified / original_data.size) * 100
            print(f"\n处理结果:")
            print(f"阈值: {threshold:.8f}")
            print(f"修改的数据点数量: {num_modified} ({percent_modified:.2f}%)")
            
            # 获取处理后数据的统计信息
            print("\n处理后数据统计:")
            print(f"最小值: {processed_data.min():.8f}")
            print(f"最大值: {processed_data.max():.8f}")
            print(f"均值: {processed_data.mean():.8f}")
            print(f"标准差: {processed_data.std():.8f}")
            
            # 如果指定了输出路径，保存处理后的MRC文件
            if output_path:
                print(f"\n保存处理后的MRC文件到: {output_path}")
                # 确保输出目录存在
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # 保存处理后的数据到新的MRC文件
                with mrcfile.new(output_path, overwrite=True) as new_mrc:
                    # 设置新数据
                    new_mrc.set_data(processed_data.astype(np.float32))
                    
                    # 复制部分原始MRC的头信息
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
            
            # 绘制原始和处理后的直方图
            if plot:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 原始数据直方图
                hist1, bins1, _ = ax1.hist(original_data.flatten(), bins=100, alpha=0.7)
                ax1.set_title('Original Data Histogram')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Frequency')
                ax1.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
                ax1.legend()
                
                # 处理后数据直方图
                ax2.hist(processed_data.flatten(), bins=100, alpha=0.7)
                ax2.set_title('Processed Data Histogram')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                
                # 保存图像
                plot_output = os.path.splitext(input_path)[0] + "_density_adjusted_plot.png"
                plt.savefig(plot_output)
                print(f"\n已保存可视化图像到: {plot_output}")
                
                # 显示图像
                plt.show()
                
                # 创建第二个图，用于显示密度分布的细节
                plt.figure(figsize=(10, 6))
                
                # 计算核密度估计
                kde_original = stats.gaussian_kde(original_data.flatten())
                kde_processed = stats.gaussian_kde(processed_data.flatten())
                
                # 创建评估点
                x_eval = np.linspace(original_data.min(), original_data.max(), 1000)
                
                # 绘制核密度估计
                plt.plot(x_eval, kde_original(x_eval), label='Original')
                plt.plot(x_eval, kde_processed(x_eval), label='Processed')
                plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
                
                plt.title('Density Distribution Comparison')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存KDE图像
                kde_plot_output = os.path.splitext(input_path)[0] + "_density_kde_plot.png"
                plt.savefig(kde_plot_output)
                print(f"已保存KDE可视化图像到: {kde_plot_output}")
                
                # 显示图像
                plt.show()
                
    except Exception as e:
        print(f"处理MRC文件时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='调整MRC文件的密度分布')
    parser.add_argument('input_path', type=str, help='输入MRC文件路径')
    parser.add_argument('--output_path', '-o', type=str, help='输出MRC文件路径（如果不指定，则自动生成）')
    parser.add_argument('--threshold', '-t', type=float, help='密度阈值，将小于等于此值的所有点设为最小值。如不指定，则自动找到密度峰值')
    parser.add_argument('--set_to_zero', '-z', action='store_true', help='设置为0而不是最小值')
    parser.add_argument('--plot', '-p', action='store_true', help='是否绘制原始和处理后的直方图')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，则自动生成一个
    if args.output_path is None:
        input_base = os.path.splitext(args.input_path)[0]
        args.output_path = f"{input_base}_peak_adjusted.mrc"
    
    # 处理MRC文件
    process_mrc_file(
        args.input_path, 
        args.output_path, 
        args.threshold, 
        not args.set_to_zero,  # 如果--set_to_zero为True，则set_to_min为False
        args.plot
    )
    
    print(f"\n脚本执行完成！")
    print(f"处理后的MRC文件已保存到: {os.path.abspath(args.output_path)}")
    
if __name__ == "__main__":
    main()
