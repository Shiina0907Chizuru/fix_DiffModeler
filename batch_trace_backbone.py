#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理密度图的骨架追踪脚本
基于trace_backbone.py，添加批处理功能
适配子目录结构和特定的信息文件格式
"""

import os
import argparse
import shutil
from ops.os_operation import mkdir
import json
import re
import csv
import glob
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='批量DiffModeler骨架追踪')
    parser.add_argument('--input_dir', type=str, required=True, help='输入密度图根目录(包含多个子目录)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    parser.add_argument('--info_file', type=str, default='C:\\Users\\Z\\Desktop\\20250315contour_levelandresolution.txt', 
                        help='包含分辨率和contour_level信息的文件')
    parser.add_argument('--config', type=str, default='config/diffmodeler.json', help='指定配置文件路径')
    parser.add_argument('--gpu', type=str, default=None, help='指定使用的GPU')
    parser.add_argument('--save_intermediate', action='store_true', help='是否保存中间结果')
    parser.add_argument('--map_file_pattern', type=str, default='EMD-*.map', help='密度图文件匹配模式')
    return parser.parse_args()

# 从trace_backbone.py复制的函数，用于解析带注释的JSON
def load_json_with_comments(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove inline comments (//...)
        content = re.sub(r'//.*?\n', '\n', content)
        # Remove trailing comments at end of lines
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove block comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse JSON
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        print("Trying alternative method...")
        
        # If the above fails, try a more manual approach
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Remove comments and trailing commas
            cleaned_lines = []
            for line in lines:
                # Remove comments
                line = re.sub(r'//.*', '', line)
                cleaned_lines.append(line)
            
            cleaned_content = ''.join(cleaned_lines)
            return json.loads(cleaned_content)
        except Exception as e2:
            print(f"Failed to parse config file with second method: {e2}")
            exit(1)

def init_save_path(origin_map_path, base_output_dir):
    # 获取文件名，不包含扩展名
    map_name = os.path.splitext(os.path.basename(origin_map_path))[0]
    map_name = map_name.replace("(","").replace(")","")
    
    # 在输出目录中创建对应的子目录
    save_path = os.path.join(base_output_dir, map_name)
    mkdir(save_path)
    return save_path, map_name

def set_up_environment(params):
    if params['resolution'] > 20:
        print("Maps with %.2f resolution is not supported! We only support maps with resolution 0-20A!" % params['resolution'])
        return None, None
    
    # Set GPU
    gpu_id = params['gpu']
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Process input map
    cur_map_path = os.path.abspath(params['F'])
    if cur_map_path.endswith(".gz"):
        from ops.os_operation import unzip_gz
        cur_map_path = unzip_gz(cur_map_path)

    # Setup output directory
    save_path, map_name = init_save_path(cur_map_path, params['output'])
    save_path = os.path.abspath(save_path)
    
    # Process map
    from data_processing.Unify_Map import Unify_Map
    cur_map_path = Unify_Map(cur_map_path, os.path.join(save_path, map_name + "_unified.mrc"))
    
    from data_processing.Resize_Map import Resize_Map
    cur_map_path = Resize_Map(cur_map_path, os.path.join(save_path, map_name + ".mrc"))
    
    # Handle negative contour level
    if params['contour'] < 0:
        from ops.map_utils import increase_map_density
        cur_map_path = increase_map_density(cur_map_path, os.path.join(save_path, map_name+"_increase.mrc"), params['contour'])
        params['contour'] = 0
    
    # Segment map - 使用传入的contour参数而不是固定为0
    from modeling.map_utils import segment_map
    new_map_path = os.path.join(save_path, map_name + "_segment.mrc")
    segment_map(cur_map_path, new_map_path, contour=params['contour'])
    
    return save_path, new_map_path

def diffusion_trace_map(save_path, cur_map_path, params):
    if params['resolution'] >= 2:
        from predict.infer_diffusion import infer_diffem
        diffusion_dir = os.path.join(save_path, "infer_diffusion")
        diff_trace_map = infer_diffem(cur_map_path, diffusion_dir, params)
    else:
        print("Skip diffusion with very high resolution map %f" % params['resolution'])
        diff_trace_map = cur_map_path
    
    print(f"Diffusion process finished! Traced map saved here: {diff_trace_map}")
    
    # Segment this difftrace map
    from modeling.map_utils import segment_map
    diff_new_trace_map = os.path.join(save_path, "diffusion.mrc")
    segment_map(diff_trace_map, diff_new_trace_map, contour=0)
    
    return diff_new_trace_map

def load_resolution_contour_info(info_file):
    """
    从信息文件中加载分辨率和contour_level信息
    返回字典: {EMD编号: (分辨率, contour_level, PDB代码)}
    """
    info_dict = {}
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                # 解析格式: "PDB代码: EMD-编号, Contour Level: 值, Resolution: 值 A"
                # 例如: "5jzh: EMD-8185, Contour Level: 0.184, Resolution: 3.9 A"
                try:
                    # 解析PDB代码和EMD编号
                    pdb_emd_match = re.match(r'([\w\d]+):\s*EMD-([\d]+)', line)
                    if not pdb_emd_match:
                        print(f"警告: 无法解析PDB代码和EMD编号: {line}")
                        continue
                    
                    pdb_code = pdb_emd_match.group(1)
                    emd_number = pdb_emd_match.group(2)
                    
                    # 解析Contour Level
                    contour_match = re.search(r'Contour\s+Level:\s*([\d\.]+)', line)
                    if not contour_match:
                        print(f"警告: 无法解析Contour Level: {line}")
                        continue
                    contour = float(contour_match.group(1))
                    
                    # 解析Resolution
                    resolution_match = re.search(r'Resolution:\s*([\d\.]+)\s*A', line)
                    if not resolution_match:
                        print(f"警告: 无法解析Resolution: {line}")
                        continue
                    resolution = float(resolution_match.group(1))
                    
                    # 保存到字典 - 使用EMD编号作为键
                    info_dict[emd_number] = (resolution, contour, pdb_code)
                    
                except Exception as e:
                    print(f"警告: 解析行时出错: {line}")
                    print(f"错误: {str(e)}")
    except Exception as e:
        print(f"加载信息文件时出错: {e}")
    
    print(f"成功加载了 {len(info_dict)} 个EMD编号的信息")
    return info_dict

def process_single_map(map_file, params, running_dir):
    """处理单个密度图文件"""
    print(f"\n正在处理: {map_file}")
    
    # 检查是否已有处理过的segment文件
    map_dir = os.path.dirname(map_file)
    processed_dir = os.path.join(map_dir, "processed")
    map_name = os.path.splitext(os.path.basename(map_file))[0]
    
    existing_segment = None
    if os.path.exists(processed_dir):
        segment_files = glob.glob(os.path.join(processed_dir, "*_segment.mrc"))
        if segment_files:
            existing_segment = segment_files[0]
            print(f"找到已有的segment文件: {existing_segment}")
    
    # 输出目录设置
    save_path, map_name = init_save_path(map_file, params['output'])
    save_path = os.path.abspath(save_path)
    
    if existing_segment:
        # 直接使用现有的segment文件，跳过预处理
        processed_map_path = os.path.join(save_path, map_name + "_segment.mrc")
        # 确保目录存在
        os.makedirs(os.path.dirname(processed_map_path), exist_ok=True)
        # 复制segment文件到输出目录
        shutil.copy(existing_segment, processed_map_path)
        print(f"使用现有的segment文件，已复制到: {processed_map_path}")
    else:
        # 没有现有文件，执行正常的预处理流程
        save_path, processed_map_path = set_up_environment(params)
        if save_path is None:
            print(f"跳过处理 {map_file}: 不支持的分辨率")
            return None
    
    # 运行扩散追踪骨架
    backbone_map_path = diffusion_trace_map(save_path, processed_map_path, params)
    
    # 复制结果到更易访问的位置
    final_output = os.path.join(save_path, "traced_backbone.mrc")
    shutil.copy(backbone_map_path, final_output)
    
    # 打印最终消息
    print(f"✅ 骨架追踪成功完成!")
    print(f"📊 原始图: {params['F']}")
    print(f"🔍 分辨率: {params['resolution']}Å")
    print(f"🧬 追踪的骨架保存至: {final_output}")
    
    return final_output

def find_map_files_in_subdirs(root_dir, pattern):
    """
    在根目录的子目录中查找匹配模式的密度图文件
    返回列表: [(emd编号, 完整文件路径, 子目录名)]
    """
    results = []
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        # 在子目录中查找匹配模式的文件
        map_files = glob.glob(os.path.join(subdir_path, pattern))
        
        for map_file in map_files:
            # 从文件名提取EMD编号
            # 例如：从 "EMD-8185.map" 提取 "8185"
            emd_match = re.search(r'EMD-([\d]+)', os.path.basename(map_file))
            if emd_match:
                emd_number = emd_match.group(1)
                results.append((emd_number, map_file, subdir))
    
    return results

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_json_with_comments(args.config)
    
    # 加载分辨率和轮廓级别信息
    info_dict = load_resolution_contour_info(args.info_file)
    
    # 获取输入目录和输出目录
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    mkdir(output_dir)
    
    # 在子目录中查找密度图文件
    map_file_infos = find_map_files_in_subdirs(input_dir, args.map_file_pattern)
    print(f"在 {input_dir} 的子目录中找到 {len(map_file_infos)} 个匹配的密度图文件")
    
    # 获取运行目录
    running_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(running_dir)
    
    # 确保模型路径是绝对路径
    if not os.path.isabs(config['model']['path']):
        config['model']['path'] = os.path.join(running_dir, config['model']['path'])
    
    # 批量处理所有密度图
    results = []
    for emd_number, map_file, subdir in tqdm(map_file_infos, desc="处理密度图"):
        # 创建参数字典，合并配置和特定参数
        params = config.copy()
        
        # 添加命令行参数
        params.update({
            'F': map_file,
            'gpu': args.gpu,
            'output': os.path.join(output_dir, subdir),  # 在输出目录中保持相同的子目录结构
            'save_intermediate': args.save_intermediate,
        })
        
        # 从信息文件获取分辨率和contour_level
        if emd_number in info_dict:
            resolution, contour, pdb_code = info_dict[emd_number]
            params['resolution'] = resolution
            params['contour'] = contour
            print(f"\n处理: {subdir} (EMD-{emd_number}, PDB: {pdb_code})")
            print(f"使用信息文件中的参数 - 分辨率: {resolution}Å, 轮廓级别: {contour}")
        else:
            # 使用默认值
            params['resolution'] = 5.0
            params['contour'] = 0.0
            print(f"\n处理: {subdir} (EMD-{emd_number})")
            print(f"警告: 在信息文件中找不到EMD-{emd_number}的信息，使用默认值 - 分辨率: 5.0Å, 轮廓级别: 0.0")
        
        # 处理单个密度图
        result = process_single_map(map_file, params, running_dir)
        if result:
            results.append((emd_number, subdir, result))
    
    # 输出处理结果摘要
    print("\n=== 批处理完成摘要 ===")
    print(f"总共处理: {len(map_file_infos)} 个文件")
    print(f"成功完成: {len(results)} 个文件")
    
    # 生成结果文件
    result_file = os.path.join(output_dir, "batch_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("EMD编号\t子目录\t输出路径\n")
        for emd_number, subdir, path in results:
            f.write(f"EMD-{emd_number}\t{subdir}\t{path}\n")
    
    print(f"结果摘要已保存至: {result_file}")

if __name__ == "__main__":
    main()
