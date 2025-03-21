#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import mrcfile

def extract_pdb_code(folder_name):
    """
    从文件夹名称中提取PDB代码
    例如：从'PDB-6j0m-EMD-9764'提取'6j0m'
    """
    match = re.search(r'PDB-(\w+)-EMD-', folder_name)
    if match:
        return match.group(1).lower()  # 确保返回小写形式
    return None

def extract_emd_code(folder_name):
    """
    从文件夹名称中提取EMD代码
    例如：从'PDB-6j0m-EMD-9764'提取'9764'
    """
    match = re.search(r'EMD-(\d+)', folder_name)
    if match:
        return match.group(1)
    return None

def find_original_map(protein_folder, pdb_code, emd_code):
    """
    在蛋白质文件夹中查找原始密度图文件
    根据提供的图片，文件可能直接位于蛋白质文件夹内
    或者在路径如 /FiniaPDB/PDB-xxxx-EMD-xxxx/EMD-xxxx.map
    """
    # 直接检查常见命名模式
    patterns = [
        os.path.join(protein_folder, f"EMD-{emd_code}.map"),
        os.path.join(protein_folder, f"emd_{emd_code}.map"),
        os.path.join(protein_folder, f"{emd_code}.map"),
        os.path.join(protein_folder, f"EMD-{emd_code}.mrc"),
        os.path.join(protein_folder, f"emd_{emd_code}.mrc"),
        os.path.join(protein_folder, f"{emd_code}.mrc"),
        os.path.join(protein_folder, f"{pdb_code}.mrc"),  # 根据图片1，有些文件可能直接用PDB代码命名
        os.path.join(protein_folder, f"{pdb_code}_increase.mrc")  # 根据图片1观察到的命名格式
    ]
    
    # 检查子文件夹内的可能位置 (根据图片2观察到的结构)
    finial_path = os.path.join(protein_folder, "EMD-" + emd_code + ".map")
    if os.path.exists(finial_path):
        return finial_path
    
    # 在FiniaPDB子目录下查找
    finial_dir = os.path.join(protein_folder, "FiniaPDB", f"PDB-{pdb_code}-EMD-{emd_code}")
    if os.path.exists(finial_dir):
        finial_path = os.path.join(finial_dir, f"EMD-{emd_code}.map")
        if os.path.exists(finial_path):
            return finial_path
    
    # 检查所有模式
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    
    # 如果没找到，搜索文件夹中所有的.map和.mrc文件
    map_files = glob.glob(os.path.join(protein_folder, "*.map"))
    map_files.extend(glob.glob(os.path.join(protein_folder, "*.mrc")))
    
    # 根据文件名包含EMD代码或只是识别正确的原始密度图
    for map_file in map_files:
        base_name = os.path.basename(map_file).lower()
        # 排除backbone、segment等非原始密度图
        if (emd_code in base_name or pdb_code in base_name) and \
           "_backbone" not in base_name and "_segment" not in base_name and \
           "_unified" not in base_name:
            return map_file
    
    # 如果还是没找到，尝试递归搜索所有子目录
    for root, dirs, files in os.walk(protein_folder):
        for file in files:
            if file.endswith(('.map', '.mrc')):
                base_name = file.lower()
                if (emd_code in base_name or f"emd-{emd_code}" in base_name.lower()) and \
                   "_backbone" not in base_name and "_segment" not in base_name:
                    return os.path.join(root, file)
    
    return None

def find_backbone_map(protein_folder, pdb_code, emd_code):
    """
    在蛋白质文件夹中查找骨架原子密度图文件
    根据提供的图片，文件可能位于 processed 子文件夹
    或路径如 /FiniaPDB/PDB-xxxx-EMD-xxxx/processed/xxxx_backbone.mrc
    """
    # 直接在protein_folder下查找 (根据图片1)
    patterns = [
        os.path.join(protein_folder, f"{pdb_code}_backbone.mrc"),
        os.path.join(protein_folder, f"{pdb_code.upper()}_backbone.mrc")
    ]
    
    # 在processed子文件夹中查找
    processed_folder = os.path.join(protein_folder, "processed")
    if os.path.exists(processed_folder):
        patterns.extend([
            os.path.join(processed_folder, f"{pdb_code}_backbone.mrc"),
            os.path.join(processed_folder, f"{pdb_code.upper()}_backbone.mrc")
        ])
    
    # 在FiniaPDB下的processed子文件夹中查找 (根据图片2)
    finial_dir = os.path.join(protein_folder, "FiniaPDB", f"PDB-{pdb_code}-EMD-{emd_code}", "processed")
    if os.path.exists(finial_dir):
        patterns.append(os.path.join(finial_dir, f"{pdb_code}_backbone.mrc"))
    
    # 检查所有模式
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    
    # 如果没找到，搜索所有可能的文件夹中包含"backbone"的文件
    backbone_search_paths = [
        protein_folder,
        processed_folder,
        finial_dir
    ]
    
    for search_path in backbone_search_paths:
        if search_path and os.path.exists(search_path):
            backbone_files = glob.glob(os.path.join(search_path, "*backbone*.mrc"))
            backbone_files.extend(glob.glob(os.path.join(search_path, "*backbone*.map")))
            if backbone_files:
                return backbone_files[0]
    
    # 如果还是没找到，递归搜索所有子目录
    for root, dirs, files in os.walk(protein_folder):
        for file in files:
            if "_backbone" in file.lower() and file.endswith(('.map', '.mrc')):
                return os.path.join(root, file)
    
    return None

def find_segment_map(protein_folder, pdb_code, emd_code):
    """
    在蛋白质文件夹中查找segment密度图文件
    类似于backbone文件的查找逻辑，但查找包含"segment"的文件
    """
    # 直接在protein_folder下查找
    patterns = [
        os.path.join(protein_folder, f"{pdb_code}_segment.mrc"),
        os.path.join(protein_folder, f"{pdb_code.upper()}_segment.mrc")
    ]
    
    # 在processed子文件夹中查找
    processed_folder = os.path.join(protein_folder, "processed")
    if os.path.exists(processed_folder):
        patterns.extend([
            os.path.join(processed_folder, f"{pdb_code}_segment.mrc"),
            os.path.join(processed_folder, f"{pdb_code.upper()}_segment.mrc")
        ])
    
    # 在FiniaPDB下的processed子文件夹中查找
    finial_dir = os.path.join(protein_folder, "FiniaPDB", f"PDB-{pdb_code}-EMD-{emd_code}", "processed")
    if os.path.exists(finial_dir):
        patterns.append(os.path.join(finial_dir, f"{pdb_code}_segment.mrc"))
    
    # 检查所有模式
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    
    # 如果没找到，搜索所有可能的文件夹中包含"segment"的文件
    segment_search_paths = [
        protein_folder,
        processed_folder,
        finial_dir
    ]
    
    for search_path in segment_search_paths:
        if search_path and os.path.exists(search_path):
            segment_files = glob.glob(os.path.join(search_path, "*segment*.mrc"))
            segment_files.extend(glob.glob(os.path.join(search_path, "*segment*.map")))
            if segment_files:
                return segment_files[0]
    
    # 如果还是没找到，递归搜索所有子目录
    for root, dirs, files in os.walk(protein_folder):
        for file in files:
            if "_segment" in file.lower() and file.endswith(('.map', '.mrc')):
                return os.path.join(root, file)
    
    return None

def calculate_map_statistics(map_file):
    """
    计算密度图的统计信息
    """
    try:
        with mrcfile.open(map_file, permissive=True) as mrc:
            data = mrc.data
            mean_value = np.mean(data)
            std_dev = np.std(data)
            min_value = np.min(data)
            max_value = np.max(data)
            
            return {
                "mean": mean_value,
                "std_dev": std_dev,
                "min_value": min_value,
                "max_value": max_value
            }
    except Exception as e:
        print(f"计算 {map_file} 的统计信息时出错: {str(e)}")
        return None

def calculate_sigma_factor(contour_level, std_dev):
    """
    计算sigma系数 = contour_level / std_dev
    """
    if std_dev == 0:
        return 0
    return contour_level / std_dev

def process_proteins(data_root, contour_level_file, output_file):
    """
    处理所有蛋白质文件夹，计算骨架原子密度图的contour level
    
    参数:
    - data_root: 包含所有蛋白质文件夹的根目录
    - contour_level_file: 包含蛋白质名称和contour level的文件
    - output_file: 输出结果的文件
    """
    # 读取contour level文件
    contour_levels = {}
    try:
        with open(contour_level_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(':')
                if len(parts) == 2:
                    protein_id = parts[0].strip()
                    try:
                        contour_level = float(parts[1].strip())
                        contour_levels[protein_id] = contour_level
                    except ValueError:
                        print(f"忽略无效的contour level: {line}")
    except Exception as e:
        print(f"读取contour level文件出错: {str(e)}")
        return
    
    print(f"成功从 {contour_level_file} 读取了 {len(contour_levels)} 个蛋白质的contour level")
    
    # 查找所有蛋白质文件夹
    protein_folders = []
    
    # 检查data_root下所有可能是蛋白质文件夹的目录
    for item in os.listdir(data_root):
        full_path = os.path.join(data_root, item)
        if os.path.isdir(full_path):
            if item.startswith("PDB-") and "EMD-" in item:
                protein_folders.append(item)
            # 也考虑其他可能的命名模式
            elif re.match(r'\d\w+', item) and os.path.exists(os.path.join(full_path, item + ".mrc")):
                # 针对图片1显示的格式，如果文件夹名为5jzh并包含5jzh.mrc
                protein_folders.append(item)
    
    print(f"在 {data_root} 中找到 {len(protein_folders)} 个蛋白质文件夹")
    
    # 如果没找到任何蛋白质文件夹，尝试直接处理data_root
    if not protein_folders and os.path.exists(os.path.join(data_root, "FiniaPDB")):
        finial_dir = os.path.join(data_root, "FiniaPDB")
        print(f"在 {data_root} 中未找到蛋白质文件夹，但找到了FiniaPDB目录，尝试直接处理")
        # 查找FiniaPDB下的所有PDB-xxx-EMD-xxx文件夹
        for item in os.listdir(finial_dir):
            full_path = os.path.join(finial_dir, item)
            if os.path.isdir(full_path) and item.startswith("PDB-") and "EMD-" in item:
                protein_folders.append(os.path.join("FiniaPDB", item))
    
    if not protein_folders:
        print(f"在 {data_root} 中未找到任何蛋白质文件夹，程序终止")
        return
    
    # 处理每个蛋白质文件夹
    results = []
    for folder in protein_folders:
        protein_folder = os.path.join(data_root, folder)
        
        # 从文件夹名提取PDB和EMD代码
        if os.path.basename(folder).startswith("PDB-"):
            pdb_code = extract_pdb_code(os.path.basename(folder))
            emd_code = extract_emd_code(os.path.basename(folder))
        else:
            # 处理像5jzh这样的文件夹名
            pdb_code = os.path.basename(folder).lower()
            emd_code = None  # 在这种情况下没有明确的EMD代码
            
            # 尝试从contour_levels找到对应的EMD代码
            for key in contour_levels.keys():
                if pdb_code in key.lower():
                    emd_code = key
                    break
        
        if not pdb_code:
            print(f"无法从文件夹名称 {folder} 提取PDB代码，跳过")
            continue
        
        print(f"\n处理蛋白质: {folder} (PDB: {pdb_code}" + (f", EMD: {emd_code})" if emd_code else ")"))
        
        # 查找原始密度图
        original_map = find_original_map(protein_folder, pdb_code, emd_code)
        if not original_map:
            print(f"在 {protein_folder} 中未找到原始密度图，跳过")
            continue
        
        print(f"找到原始密度图: {original_map}")
        
        # 查找骨架原子密度图
        backbone_map = find_backbone_map(protein_folder, pdb_code, emd_code)
        if not backbone_map:
            print(f"未找到 {pdb_code} 的骨架原子密度图，跳过")
            continue
        
        print(f"找到骨架原子密度图: {backbone_map}")
        
        # 查找segment密度图
        segment_map = find_segment_map(protein_folder, pdb_code, emd_code)
        if segment_map:
            print(f"找到segment密度图: {segment_map}")
        else:
            print(f"未找到 {pdb_code} 的segment密度图，将忽略segment处理")
        
        # 获取该蛋白质的contour level
        contour_level = None
        
        # 尝试不同的键名查找contour level
        possible_keys = [
            emd_code,
            f"EMD-{emd_code}" if emd_code else None,
            pdb_code,
            f"{pdb_code.upper()}"
        ]
        
        for key in possible_keys:
            if key and key in contour_levels:
                contour_level = contour_levels[key]
                print(f"找到contour level: {contour_level} (键: {key})")
                break
        
        if contour_level is None:
            print(f"在contour level文件中未找到蛋白质 {pdb_code} 或 EMD-{emd_code} 的信息，跳过")
            continue
        
        # 计算原始密度图的统计信息
        original_stats = calculate_map_statistics(original_map)
        if not original_stats:
            print(f"计算原始密度图统计信息失败，跳过")
            continue
        
        # 计算sigma系数
        sigma_factor = calculate_sigma_factor(contour_level, original_stats["std_dev"])
        print(f"原始密度图标准差: {original_stats['std_dev']:.6f}, Sigma系数: {sigma_factor:.6f}")
        
        # 计算骨架原子密度图的统计信息
        backbone_stats = calculate_map_statistics(backbone_map)
        if not backbone_stats:
            print(f"计算骨架原子密度图统计信息失败，跳过")
            continue
        
        # 计算骨架原子密度图的contour level
        backbone_contour = sigma_factor * backbone_stats["std_dev"]
        print(f"骨架原子密度图标准差: {backbone_stats['std_dev']:.6f}, 计算得到的contour level: {backbone_contour:.6f}")
        
        # 计算segment密度图的统计信息和contour level（如果存在）
        segment_stats = None
        segment_contour = None
        if segment_map:
            segment_stats = calculate_map_statistics(segment_map)
            if segment_stats:
                segment_contour = sigma_factor * segment_stats["std_dev"]
                print(f"Segment密度图标准差: {segment_stats['std_dev']:.6f}, 计算得到的contour level: {segment_contour:.6f}")
            else:
                print(f"计算segment密度图统计信息失败")
        
        # 保存结果
        result = {
            "protein_folder": folder,
            "pdb_code": pdb_code,
            "emd_code": emd_code if emd_code else "N/A",
            "original_map": os.path.basename(original_map),
            "original_contour": contour_level,
            "original_std_dev": original_stats["std_dev"],
            "sigma_factor": sigma_factor,
            "backbone_map": os.path.basename(backbone_map),
            "backbone_std_dev": backbone_stats["std_dev"],
            "backbone_contour": backbone_contour,
            "segment_map": os.path.basename(segment_map) if segment_map else "N/A",
            "segment_std_dev": segment_stats["std_dev"] if segment_stats else 0.0,
            "segment_contour": segment_contour if segment_contour else 0.0
        }
        
        results.append(result)
    
    # 将结果写入CSV文件
    csv_output = output_file + ".csv"
    if results:
        with open(csv_output, 'w') as f:
            f.write("protein_folder,pdb_code,emd_code,original_map,original_contour,original_std_dev,sigma_factor,"
                   "backbone_map,backbone_std_dev,backbone_contour,segment_map,segment_std_dev,segment_contour\n")
            for result in results:
                f.write(f"{result['protein_folder']},{result['pdb_code']},{result['emd_code']},{result['original_map']},"
                        f"{result['original_contour']:.6f},{result['original_std_dev']:.6f},{result['sigma_factor']:.6f},"
                        f"{result['backbone_map']},{result['backbone_std_dev']:.6f},{result['backbone_contour']:.6f},"
                        f"{result['segment_map']},{result['segment_std_dev']:.6f},{result['segment_contour']:.6f}\n")
        
        print(f"\n成功处理了 {len(results)} 个蛋白质，CSV结果已保存到: {csv_output}")
        
        # 创建一个完整信息的TXT文件 (calculate_contour.txt)
        txt_output = "calculate_contour.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write("骨架原子和Segment密度图Contour Level计算结果\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
                f.write(f"蛋白质: {result['protein_folder']}\n")
                f.write(f"PDB代码: {result['pdb_code']}, EMD代码: {result['emd_code']}\n")
                f.write(f"原始密度图: {result['original_map']}\n")
                f.write(f"    Contour Level: {result['original_contour']:.6f}\n")
                f.write(f"    标准差: {result['original_std_dev']:.6f}\n")
                f.write(f"    Sigma系数: {result['sigma_factor']:.6f}\n")
                f.write(f"骨架原子密度图: {result['backbone_map']}\n")
                f.write(f"    标准差: {result['backbone_std_dev']:.6f}\n")
                f.write(f"    计算得到的Contour Level: {result['backbone_contour']:.6f}\n")
                if result['segment_map'] != "N/A":
                    f.write(f"Segment密度图: {result['segment_map']}\n")
                    f.write(f"    标准差: {result['segment_std_dev']:.6f}\n")
                    f.write(f"    计算得到的Contour Level: {result['segment_contour']:.6f}\n")
                f.write("\n" + "-" * 40 + "\n\n")
            
            f.write("\n总结: 成功处理了 {} 个蛋白质".format(len(results)))
        
        print(f"同时创建了完整的结果文件: {txt_output}")
        
        # 创建仅包含骨架原子密度图contour level的简化TXT文件 (backbone_contourlevels.txt)
        backbone_output = "backbone_contourlevels.txt"
        with open(backbone_output, 'w', encoding='utf-8') as f:
            # 按照PDB代码排序
            sorted_results = sorted(results, key=lambda x: x['pdb_code'])
            for result in sorted_results:
                f.write(f"{result['pdb_code']}: {result['backbone_contour']:.6f}\n")
        
        print(f"创建了骨架原子密度图contour level的简化文件: {backbone_output}")
        
        # 创建仅包含segment密度图contour level的简化TXT文件 (segment_contourlevels.txt)
        segment_output = "segment_contourlevels.txt"
        with open(segment_output, 'w', encoding='utf-8') as f:
            # 按照PDB代码排序，只包含有segment数据的结果
            sorted_results = sorted([r for r in results if r['segment_map'] != "N/A"], key=lambda x: x['pdb_code'])
            for result in sorted_results:
                f.write(f"{result['pdb_code']}: {result['segment_contour']:.6f}\n")
        
        print(f"创建了segment密度图contour level的简化文件: {segment_output}")
    else:
        print("没有成功处理任何蛋白质")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算骨架原子密度图和segment密度图的contour level")
    
    # 添加参数
    parser.add_argument("--data_root", "-d", required=True, help="包含所有蛋白质文件夹的根目录")
    parser.add_argument("--contour_file", "-c", required=True, help="包含蛋白质名称和contour level的文件")
    parser.add_argument("--output", "-o", default="contour_levels", help="输出结果的基础文件名（不含扩展名）")
    
    # 解析参数
    args = parser.parse_args()
    
    # 处理蛋白质
    process_proteins(args.data_root, args.contour_file, args.output)

if __name__ == "__main__":
    main()
