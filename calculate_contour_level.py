#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import mrcfile

def calculate_contour_level(mrc_path, resolution=None, sigma_multiplier=None, show_multiple_sigmas=False):
    """
    计算MRC文件的合适contour level值。
    
    参数:
    - mrc_path (str): MRC文件路径
    - resolution (float): 电镜数据的分辨率，单位为埃(Å)
    - sigma_multiplier (float): 手动指定sigma倍数，如果提供则忽略resolution
    - show_multiple_sigmas (bool): 是否显示多个sigma倍数下的contour level
    
    返回:
    - dict: 包含计算结果的字典
    """
    # 检查文件是否存在
    if not os.path.exists(mrc_path):
        raise FileNotFoundError(f"MRC文件 {mrc_path} 不存在")
    
    # 打开MRC文件
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        # 获取数据
        data = mrc.data
        # 计算统计值
        mean_value = np.mean(data)
        std_dev = np.std(data)
        min_value = np.min(data)
        max_value = np.max(data)
        
        # 确定合适的sigma倍数
        if sigma_multiplier is None:
            if resolution is None:
                # 默认值
                sigma_multiplier = 1.5
                print(f"未提供分辨率或sigma倍数，使用默认sigma倍数: {sigma_multiplier}")
            else:
                # 根据分辨率选择合适的sigma倍数
                if resolution <= 2.0:  # 高分辨率
                    sigma_multiplier = 2.0
                elif resolution <= 3.5:  # 中等分辨率
                    sigma_multiplier = 1.5
                elif resolution <= 5.0:  # 低分辨率
                    sigma_multiplier = 1.0
                else:  # 非常低的分辨率
                    sigma_multiplier = 0.8
                
                print(f"根据分辨率 {resolution}Å 选择sigma倍数: {sigma_multiplier}")
        else:
            print(f"使用手动指定的sigma倍数: {sigma_multiplier}")
        
        # 计算contour level
        contour_level = mean_value + sigma_multiplier * std_dev
        
        # 处理边界情况
        if contour_level > max_value:
            print(f"警告: 计算出的contour level({contour_level})大于最大值({max_value})，将使用最大值的90%")
            contour_level = max_value * 0.9
        
        # 准备结果
        results = {
            "file_name": os.path.basename(mrc_path),
            "mean": mean_value,
            "std_dev": std_dev,
            "min_value": min_value,
            "max_value": max_value,
            "sigma_multiplier": sigma_multiplier,
            "contour_level": contour_level,
            "resolution": resolution
        }
        
        # 如果需要显示多个sigma倍数下的结果
        if show_multiple_sigmas:
            sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
            results["multiple_sigmas"] = {}
            for sigma in sigma_values:
                level = mean_value + sigma * std_dev
                if level <= max_value:
                    results["multiple_sigmas"][sigma] = level
        
        return results

def print_results(results, show_multiple_sigmas=False):
    """打印计算结果"""
    print("\n----- MRC文件统计信息 -----")
    print(f"文件名: {results['file_name']}")
    if results['resolution'] is not None:
        print(f"分辨率: {results['resolution']}Å")
    print(f"均值: {results['mean']:.6f}")
    print(f"标准差: {results['std_dev']:.6f}")
    print(f"最小值: {results['min_value']:.6f}")
    print(f"最大值: {results['max_value']:.6f}")
    print(f"选用的Sigma倍数: {results['sigma_multiplier']}")
    print(f"计算的Contour Level: {results['contour_level']:.6f}")
    print(f"推荐的显示值: {results['contour_level']:.6f}")
    
    # 如果需要显示多个sigma倍数下的结果
    if show_multiple_sigmas and "multiple_sigmas" in results:
        print("\n----- 不同Sigma倍数下的Contour Level值 -----")
        print(f"{'Sigma倍数':<10} {'Contour Level':<15} {'说明':<20}")
        for sigma, level in sorted(results["multiple_sigmas"].items()):
            description = ""
            if sigma == 1.0:
                description = "显示更多细节"
            elif sigma == 1.5:
                description = "标准设置"
            elif sigma == 3.0:
                description = "高置信度特征"
            elif sigma >= 6.0:
                description = "非常高置信度特征"
            
            print(f"{sigma:<10} {level:<15.6f} {description:<20}")
    
    print("---------------------------\n")

def batch_process_directory(directory_path, output_file=None, resolution=None, sigma_multiplier=None, show_multiple_sigmas=False):
    """
    批处理目录中的所有MRC文件
    
    参数:
    - directory_path (str): 目录路径
    - output_file (str): 输出文件路径
    - resolution (float): 所有MRC文件的分辨率
    - sigma_multiplier (float): 手动指定sigma倍数
    - show_multiple_sigmas (bool): 是否显示多个sigma倍数下的contour level
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"{directory_path} 不是一个有效的目录")
    
    # 收集所有MRC文件
    mrc_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.mrc', '.map')):
                mrc_files.append(os.path.join(root, file))
    
    if not mrc_files:
        print(f"在 {directory_path} 中没有找到MRC文件")
        return
    
    print(f"在 {directory_path} 中找到 {len(mrc_files)} 个MRC文件")
    
    # 处理所有文件并收集结果
    all_results = []
    for mrc_file in mrc_files:
        try:
            print(f"处理: {mrc_file}")
            result = calculate_contour_level(mrc_file, resolution, sigma_multiplier, show_multiple_sigmas)
            print_results(result, show_multiple_sigmas)
            all_results.append(result)
        except Exception as e:
            print(f"处理 {mrc_file} 时出错: {str(e)}")
    
    # 如果指定了输出文件，将结果写入文件
    if output_file and all_results:
        with open(output_file, 'w') as f:
            f.write("file_name,mean,std_dev,min_value,max_value,sigma_multiplier,contour_level,resolution\n")
            for result in all_results:
                resolution_str = str(result['resolution']) if result['resolution'] is not None else "NA"
                f.write(f"{result['file_name']},{result['mean']:.6f},{result['std_dev']:.6f},"
                        f"{result['min_value']:.6f},{result['max_value']:.6f},{result['sigma_multiplier']},"
                        f"{result['contour_level']:.6f},{resolution_str}\n")
        
        print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算MRC文件的合适contour level值")
    
    # 添加参数
    parser.add_argument("input", help="MRC文件路径或包含MRC文件的目录")
    parser.add_argument("--resolution", "-r", type=float, help="电镜数据的分辨率，单位为埃(Å)")
    parser.add_argument("--sigma", "-s", type=float, help="手动指定sigma倍数，优先级高于分辨率自动选择")
    parser.add_argument("--output", "-o", help="保存结果的CSV文件路径(仅在处理目录时有效)")
    parser.add_argument("--show-all-sigmas", "-a", action="store_true", help="显示多个sigma倍数下的contour level值")
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        try:
            results = calculate_contour_level(args.input, args.resolution, args.sigma, args.show_all_sigmas)
            print_results(results, args.show_all_sigmas)
        except Exception as e:
            print(f"错误: {str(e)}")
    elif os.path.isdir(args.input):
        # 处理目录
        try:
            batch_process_directory(args.input, args.output, args.resolution, args.sigma, args.show_all_sigmas)
        except Exception as e:
            print(f"错误: {str(e)}")
    else:
        print(f"错误: {args.input} 既不是文件也不是目录")

if __name__ == "__main__":
    main()
