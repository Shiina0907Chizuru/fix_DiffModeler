import os
import mrcfile
import numpy as np
import argparse
from pathlib import Path


def read_contour_file(file_path):
    """读取contour文件，提取蛋白质名称、sigma值和contour值"""
    proteins_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    current_protein = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测新的蛋白质记录开始
        if line.startswith("蛋白质:") or line.startswith("PDB代码:"):
            if current_protein is not None:
                proteins_data.append(current_protein)
            
            # 解析蛋白质名称
            protein_name = None
            if line.startswith("蛋白质:"):
                parts = line.split("PDB-")
                if len(parts) > 1:
                    emd_parts = parts[1].split("-EMD-")
                    if len(emd_parts) > 0:
                        protein_name = emd_parts[0]
            elif line.startswith("PDB代码:"):
                parts = line.split("PDB代码:")
                if len(parts) > 1:
                    protein_name = parts[1].split(",")[0].strip()
            
            if protein_name:
                current_protein = {
                    "name": protein_name,
                    "path": "",
                    "backbone": {},
                    "segment": {}
                }
        
        # 对当前蛋白质，提取骨架密度图信息
        if current_protein and "骨架原子密度图" in line:
            j = i + 1
            while j < len(lines) and "标准差" in lines[j]:
                current_protein["backbone"]["std"] = float(lines[j].split(":")[1].strip())
                j += 1
                
            while j < len(lines) and "计算得到的Contour" in lines[j]:
                current_protein["backbone"]["contour"] = float(lines[j].split(":")[1].strip())
                j += 1
        
        # 提取segment密度图信息
        if current_protein and "Segment密度图" in line:
            j = i + 1
            while j < len(lines) and "标准差" in lines[j]:
                current_protein["segment"]["std"] = float(lines[j].split(":")[1].strip())
                j += 1
                
            while j < len(lines) and "计算得到的Contour" in lines[j]:
                current_protein["segment"]["contour"] = float(lines[j].split(":")[1].strip())
                j += 1
            
        # 提取sigma值
        if current_protein and "Sigma系数" in line:
            current_protein["backbone"]["sigma"] = float(line.split(":")[1].strip())
        
        # 提取原始contour值
        if current_protein and "Contour Level" in line:
            current_protein["segment"]["orig_contour"] = float(line.split(":")[1].strip())
            
        i += 1
    
    # 添加最后一个蛋白质
    if current_protein is not None:
        proteins_data.append(current_protein)
    
    return proteins_data


def find_density_maps(protein_path, protein_name):
    """查找蛋白质文件夹中的backbone和segment密度图"""
    backbone_map = None
    segment_map = None
    
    # 从图片中可以看出，文件在processed目录下
    processed_dir = os.path.join(protein_path, "processed")
    
    if os.path.exists(processed_dir):
        # 查找backbone密度图
        backbone_file = f"{protein_name}_backbone.mrc"
        backbone_path = os.path.join(processed_dir, backbone_file)
        if os.path.exists(backbone_path):
            backbone_map = backbone_path
        
        # 查找segment密度图
        segment_file = f"{protein_name}_segment.mrc"
        segment_path = os.path.join(processed_dir, segment_file)
        if os.path.exists(segment_path):
            segment_map = segment_path
    
    return backbone_map, segment_map


def calculate_contour_for_one_percent(map_path):
    """计算包含1%网格点的contour level"""
    if not os.path.exists(map_path):
        print(f"警告：找不到密度图 {map_path}")
        return None
    
    try:
        with mrcfile.open(map_path, permissive=True) as mrc:
            data = mrc.data
            # 对数据排序
            sorted_data = np.sort(data.flatten())
            # 计算99百分位数的索引(从高到低包含1%的点)
            percentile_99_idx = int(len(sorted_data) * 0.99)
            # 获取对应的contour值
            contour_level = sorted_data[percentile_99_idx]
            print(f"计算得到的1%网格点contour level: {contour_level}")
            return contour_level
    except Exception as e:
        print(f"计算1%网格点contour level时出错 {map_path}: {str(e)}")
        return None


def adjust_contour_if_needed(map_path, contour_value, threshold_percentage=0.9):
    """如果大于contour的体素比例超过阈值，将contour乘以1.5"""
    if not os.path.exists(map_path):
        print(f"警告：找不到密度图 {map_path}")
        return contour_value, 0, 0
    
    try:
        with mrcfile.open(map_path, permissive=True) as mrc:
            data = mrc.data
            total_voxels = data.size
            voxels_above_contour = np.sum(data > contour_value)
            percentage = voxels_above_contour / total_voxels
            
            adjusted_contour = contour_value
            if percentage > threshold_percentage:
                adjusted_contour = contour_value * 1.5
                print(f"调整contour: {contour_value} -> {adjusted_contour} (超过阈值比例: {percentage:.2f})")
            
            return adjusted_contour, voxels_above_contour, total_voxels
    except Exception as e:
        print(f"处理密度图时出错 {map_path}: {str(e)}")
        return contour_value, 0, 0


def compare_density_maps(backbone_map, segment_map, backbone_contour, segment_contour, overlap_threshold=0.95):
    """比较backbone和segment密度图，统计重叠率"""
    if not os.path.exists(backbone_map) or not os.path.exists(segment_map):
        print(f"警告：密度图文件不存在")
        return False, 0, 0, 0
    
    try:
        with mrcfile.open(backbone_map, permissive=True) as mrc_backbone:
            backbone_data = mrc_backbone.data
        
        with mrcfile.open(segment_map, permissive=True) as mrc_segment:
            segment_data = mrc_segment.data
        
        # 检查数据形状是否相同
        if backbone_data.shape != segment_data.shape:
            print(f"警告：密度图形状不匹配 - backbone: {backbone_data.shape}, segment: {segment_data.shape}")
            return False, 0, 0, 0
        
        # 计算各种统计量
        backbone_above_contour = np.sum(backbone_data > backbone_contour)
        segment_above_contour = np.sum(segment_data > segment_contour)
        both_above_contour = np.sum((backbone_data > backbone_contour) & (segment_data > segment_contour))
        
        max_above_contour = max(backbone_above_contour, segment_above_contour)
        
        if max_above_contour == 0:
            overlap_ratio = 0
        else:
            overlap_ratio = both_above_contour / max_above_contour
        
        below_threshold = overlap_ratio < overlap_threshold
        
        return below_threshold, overlap_ratio, both_above_contour, max_above_contour
    
    except Exception as e:
        print(f"比较密度图时出错: {str(e)}")
        return False, 0, 0, 0


def main(contour_file, output_file, base_dir="/zhaoxuanj/FinialPDB", threshold_percentage=0.9, overlap_threshold=0.95):
    """主函数：处理所有蛋白质的密度图，标记符合条件的蛋白质"""
    proteins_data = read_contour_file(contour_file)
    print(f"从文件中读取了 {len(proteins_data)} 个蛋白质记录")
    
    problematic_proteins = []
    
    # 检查输出文件路径是否为目录
    if os.path.isdir(output_file):
        # 如果是目录，则在目录中创建一个默认文件名
        output_file = os.path.join(output_file, "density_analysis_result.csv")
        print(f"指定的输出路径是一个目录，将使用默认文件名: {output_file}")
    
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    with open(output_file, 'w') as out_f:
        out_f.write("蛋白质名称,重叠率,两者都大于Contour的体素数,各自大于Contour的最大体素数,Backbone调整后Contour,Segment调整后Contour\n")
        
        for protein in proteins_data:
            protein_name = protein["name"]
            
            # 构建蛋白质路径
            protein_folder = f"PDB-{protein_name.lower()}-EMD-*"  # 使用通配符匹配EMD编号
            import glob
            matching_folders = glob.glob(os.path.join(base_dir, protein_folder))
            
            if not matching_folders:
                print(f"警告：找不到匹配的蛋白质文件夹: {protein_name}")
                continue
            
            protein_path = matching_folders[0]  # 使用第一个匹配的文件夹
            print(f"\n处理蛋白质: {protein_name}")
            print(f"蛋白质路径: {protein_path}")
            
            # 找到密度图文件
            backbone_map, segment_map = find_density_maps(protein_path, protein_name)
            
            if not backbone_map or not segment_map:
                print(f"警告：找不到完整的密度图文件")
                continue
            
            print(f"Backbone密度图: {backbone_map}")
            print(f"Segment密度图: {segment_map}")
            
            # 获取原始contour值 (仅用于segment)
            if "contour" not in protein["segment"]:
                print(f"警告：缺少segment contour信息")
                continue
            
            segment_contour = protein["segment"]["contour"]
            
            # 使用新方法为backbone计算contour level，设置为包含1%的网格点
            backbone_contour = calculate_contour_for_one_percent(backbone_map)
            if backbone_contour is None:
                print(f"警告：无法计算backbone的1%网格点contour level")
                # 如果新方法失败，回退使用原方法（已注释）
                # backbone_contour = protein["backbone"]["contour"]
                continue
                
            # 打印对比信息（如果有原始的contour值）
            if "contour" in protein["backbone"]:
                print(f"Backbone Contour - 原始: {protein['backbone']['contour']}, 新方法(1%网格点): {backbone_contour}")
            
            # 根据需要调整contour值（仅对segment应用）
            # 注意: 我们不再调整backbone的contour，而是直接使用1%网格点的contour值
            # adjusted_backbone_contour, backbone_voxels_above, backbone_total = adjust_contour_if_needed(
            #    backbone_map, backbone_contour, threshold_percentage)
            adjusted_backbone_contour = backbone_contour
            
            adjusted_segment_contour, segment_voxels_above, segment_total = adjust_contour_if_needed(
                segment_map, segment_contour, threshold_percentage)
            
            # 比较密度图
            is_problematic, overlap_ratio, both_above_contour, max_above_contour = compare_density_maps(
                backbone_map, segment_map, adjusted_backbone_contour, adjusted_segment_contour, overlap_threshold)
            
            # 输出结果
            print(f"Backbone Contour (1%网格点): {backbone_contour}")
            print(f"Segment Contour: {segment_contour} -> {adjusted_segment_contour}")
            print(f"重叠率: {overlap_ratio:.4f}")
            print(f"两者都大于Contour的体素数: {both_above_contour}")
            print(f"各自大于Contour的最大体素数: {max_above_contour}")
            
            # 记录到CSV文件
            out_f.write(f"{protein_name},{overlap_ratio:.4f},{both_above_contour},{max_above_contour},{adjusted_backbone_contour},{adjusted_segment_contour}\n")
            
            # 标记问题蛋白质
            if is_problematic:
                problematic_proteins.append({
                    "name": protein_name,
                    "overlap_ratio": overlap_ratio
                })
                print(f"!!! 标记为问题蛋白质 !!!")
    
    # 输出所有问题蛋白质
    print(f"\n\n发现 {len(problematic_proteins)} 个问题蛋白质（重叠率低于 {overlap_threshold}）:")
    for protein in problematic_proteins:
        print(f"{protein['name']}: 重叠率 {protein['overlap_ratio']:.4f}")
    
    print(f"\n分析完成。结果已保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析backbone和segment密度图的重叠情况")
    parser.add_argument("--contour_file", type=str, default="C:\\Users\\Z\\Desktop\\calculate_contour.txt", 
                        help="包含蛋白质contour信息的文件路径")
    parser.add_argument("--output_file", type=str, default="C:\\Users\\Z\\Desktop\\density_analysis_result.csv", 
                        help="输出CSV文件的路径")
    parser.add_argument("--base_dir", type=str, default="/zhaoxuanj/FinialPDB", 
                        help="蛋白质文件的基础目录")
    parser.add_argument("--threshold_percentage", type=float, default=0.9, 
                        help="调整contour值的体素比例阈值")
    parser.add_argument("--overlap_threshold", type=float, default=0.95, 
                        help="标记问题蛋白质的重叠率阈值")
    
    args = parser.parse_args()
    
    main(
        args.contour_file, 
        args.output_file, 
        args.base_dir, 
        args.threshold_percentage, 
        args.overlap_threshold
    )
