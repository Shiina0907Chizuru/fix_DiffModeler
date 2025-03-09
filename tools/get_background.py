import os
import numpy as np
import mrcfile
import argparse
from tqdm import tqdm
import glob

def find_background_value(data, bins=10000):
    """
    直接在原始数据上找到最频繁的值
    
    Args:
        data: 原始数据数组
        bins: 直方图的分箱数量
    
    Returns:
        tuple: (most_frequent_value, percentage)
            - most_frequent_value: 最频繁出现的值
            - percentage: 该值占总体的百分比
    """
    # 获取数据的范围
    min_val = np.min(data)
    max_val = np.max(data)
    print(f"  Raw data stats - Min: {min_val}, Max: {max_val}, Mean: {np.mean(data)}")
    
    # 如果数据全部相同，直接返回该值
    if max_val == min_val:
        return min_val, 100.0
    
    # 创建直方图
    hist, bin_edges = np.histogram(data, bins=bins)
    
    # 找到频率最高的分箱
    max_bin_index = np.argmax(hist)
    
    # 计算分箱中心值（最频繁值）
    bin_center = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    
    # 计算该值的占比百分比
    percentage = (hist[max_bin_index] / np.sum(hist)) * 100
    
    # 打印直方图诊断信息
    print(f"  Histogram diagnostics - Total bins: {bins}, Max bin: {max_bin_index}")
    print(f"  Bin range: [{bin_edges[max_bin_index]:.8f}, {bin_edges[max_bin_index+1]:.8f}]")
    print(f"  Top 3 most frequent bins:")
    
    # 获取前3个最频繁的分箱
    top_indices = np.argsort(hist)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        bin_val = (bin_edges[idx] + bin_edges[idx+1]) / 2
        bin_pct = (hist[idx] / np.sum(hist)) * 100
        print(f"    #{i+1}: Value = {bin_val:.8f}, Percentage = {bin_pct:.2f}%")
    
    return bin_center, percentage

def normalize_value(value, min_val, max_val):
    """
    将单个值根据给定的最小值和最大值归一化到[0,1]范围
    
    Args:
        value: 要归一化的值
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        float: 归一化后的值
    """
    if max_val == min_val:
        return 0.0
    
    return (value - min_val) / (max_val - min_val)

def find_top_density(data, percentile=0.98):
    """
    找到数据的指定百分位值
    
    Args:
        data: 数据数组
        percentile: 百分位，默认0.98
        
    Returns:
        float: 对应百分位的值
    """
    # 创建直方图
    hist, bin_edges = np.histogram(data, bins=1000)
    # 计算累积分布
    cumulative_hist = np.cumsum(hist) / np.sum(hist)
    # 找到对应百分位的索引
    percentile_idx = np.argmax(cumulative_hist >= percentile)
    # 返回对应的值
    return bin_edges[percentile_idx]

def process_segment_mrc_file(mrc_path):
    """
    处理输入(segment)MRC文件找到背景值，使用与generate_infer_data一致的归一化方式
    
    Args:
        mrc_path: MRC文件路径
    
    Returns:
        tuple: (raw_background, normalized_background, percentage) 或 (None, None, None)
    """
    try:
        print(f"  处理输入文件: {mrc_path}")
        # 打开MRC文件
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            # 获取数据
            data = np.array(mrc.data)
            
            # 打印数据类型和形状
            print(f"  MRC data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Value counts: Zeros: {np.sum(data == 0)}, Non-zeros: {np.sum(data != 0)}")
            
            # 检查唯一值
            unique_vals = np.unique(data)
            print(f"  Number of unique values: {len(unique_vals)}")
            if len(unique_vals) < 10:
                print(f"  All unique values: {unique_vals}")
            else:
                print(f"  Sample unique values: {unique_vals[:5]}...{unique_vals[-5:]}")
            
            # 打印原始数据统计信息
            min_orig = np.min(data)
            max_orig = np.max(data)
            print(f"  原始数据统计: 最小值={min_orig:.8f}, 最大值={max_orig:.8f}")
            
            # 找到最频繁的背景值 (在原始数据上)
            background, percentage = find_background_value(data)
            
            # 使用与generate_infer_data一致的归一化
            # 找到98百分位值并修剪
            percentile_98 = find_top_density(data, 0.98)
            print(f"  98th percentile value: {percentile_98:.8f}")
            
            # 复制数据并修剪
            data_norm = data.copy()
            data_norm[data_norm > percentile_98] = percentile_98
            
            # 进行最小最大归一化
            min_value = np.min(data_norm)
            max_value = np.max(data_norm)
            normalized_data = (data_norm - min_value) / (max_value - min_value)
            
            # 将原始背景值归一化
            if background > percentile_98:
                # 如果背景值大于98百分位，先截断
                normalized_background = normalize_value(percentile_98, min_value, max_value)
                print(f"  背景值({background:.8f})超过98百分位，使用截断值({percentile_98:.8f})")
            else:
                normalized_background = normalize_value(background, min_value, max_value)
            
            print(f"  原始背景值: {background:.8f}, 归一化后: {normalized_background:.8f}, 占比: {percentage:.2f}%")
            
            return background, normalized_background, percentage
    except Exception as e:
        print(f"Error processing {mrc_path}: {str(e)}")
        return None, None, None

def process_backbone_mrc_file(mrc_path):
    """
    处理输出(backbone)MRC文件找到背景值
    
    Args:
        mrc_path: MRC文件路径
    
    Returns:
        tuple: (raw_background, normalized_background, percentage) 或 (None, None, None)
    """
    try:
        print(f"  处理输出文件: {mrc_path}")
        # 打开MRC文件
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            # 获取数据
            data = mrc.data
            
            # 打印数据类型和形状
            print(f"  MRC data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Value counts: Zeros: {np.sum(data == 0)}, Non-zeros: {np.sum(data != 0)}")
            
            # 检查唯一值
            unique_vals = np.unique(data)
            print(f"  Number of unique values: {len(unique_vals)}")
            if len(unique_vals) < 10:
                print(f"  All unique values: {unique_vals}")
            else:
                print(f"  Sample unique values: {unique_vals[:5]}...{unique_vals[-5:]}")
            
            # 找到最频繁的背景值
            background, percentage = find_background_value(data)
            
            # 获取数据范围用于归一化
            min_val = np.min(data)
            max_val = np.max(data)
            
            # 将背景值归一化到[0,1]范围
            normalized_background = normalize_value(background, min_val, max_val)
            print(f"  原始背景值: {background:.8f}, 归一化后: {normalized_background:.8f}, 占比: {percentage:.2f}%")
            
            return background, normalized_background, percentage
    except Exception as e:
        print(f"Error processing {mrc_path}: {str(e)}")
        return None, None, None

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Find background values in protein backbone and segment MRC files')
    parser.add_argument('--protein_list', type=str, required=True,
                        help='Path to a text file containing protein IDs, one per line')
    parser.add_argument('--base_dir', type=str, 
                        default='/defaultShare/zcan-library/Diffmodeler_data/20250306dataset/origin',
                        help='Base directory containing the protein folders')
    parser.add_argument('--output_file', type=str, default='background.txt',
                        help='Output file to write background values')
    parser.add_argument('--bins', type=int, default=10000,
                        help='Number of bins to use for the histogram')
    
    args = parser.parse_args()
    
    # 读取蛋白质ID列表
    try:
        with open(args.protein_list, 'r') as f:
            protein_ids = [line.strip() for line in f.readlines()]
        print(f"Read {len(protein_ids)} protein IDs from {args.protein_list}")
    except Exception as e:
        print(f"Error reading protein list: {str(e)}")
        return
    
    # 存储背景值的字典
    background_values = {}
    
    # 处理每个蛋白质
    print("Processing protein MRC files...")
    for protein_id in tqdm(protein_ids):
        print(f"\nProcessing protein: {protein_id}")
        # 构建MRC文件路径
        mrc_folder = os.path.join(args.base_dir, f"PDB-{protein_id.lower()}-EMD-*")
        mrc_folders = glob.glob(mrc_folder)
        
        if not mrc_folders:
            print(f"Warning: No matching folder found for protein {protein_id}")
            continue
        
        # 使用第一个匹配的文件夹
        mrc_folder = mrc_folders[0]
        
        # 构建backbone和segment MRC文件路径
        backbone_path = os.path.join(mrc_folder, "processed", f"{protein_id.lower()}_backbone.mrc")
        segment_path = os.path.join(mrc_folder, "processed", f"{protein_id.lower()}_segment.mrc")
        
        if not os.path.exists(backbone_path):
            print(f"Warning: Backbone MRC file not found at {backbone_path}")
            continue
            
        if not os.path.exists(segment_path):
            print(f"Warning: Segment MRC file not found at {segment_path}")
            continue
        
        # 处理输入(segment)MRC文件
        input_raw_bg, input_norm_bg, input_percentage = process_segment_mrc_file(segment_path)
        
        # 处理输出(backbone)MRC文件
        output_raw_bg, output_norm_bg, output_percentage = process_backbone_mrc_file(backbone_path)
        
        if input_raw_bg is not None and output_raw_bg is not None:
            background_values[protein_id] = (
                input_raw_bg, input_norm_bg, input_percentage,
                output_raw_bg, output_norm_bg, output_percentage
            )
            print(f"  RESULT: Protein {protein_id}:")
            print(f"    Input - Raw BG: {input_raw_bg:.8f}, Normalized: {input_norm_bg:.8f}, Percentage: {input_percentage:.2f}%")
            print(f"    Output - Raw BG: {output_raw_bg:.8f}, Normalized: {output_norm_bg:.8f}, Percentage: {output_percentage:.2f}%")
    
    # 将背景值写入文件
    try:
        with open(args.output_file, 'w') as f:
            f.write("# protein_id inputbackground inputpercent outputbackground outputpercent\n")
            for protein_id, values in background_values.items():
                input_raw_bg, input_norm_bg, input_percentage, output_raw_bg, output_norm_bg, output_percentage = values
                f.write(f"{protein_id} inputbackground: {input_norm_bg:.8f} inputpercent: {input_percentage:.2f} "
                        f"outputbackground: {output_norm_bg:.8f} outputpercent: {output_percentage:.2f}\n")
        print(f"Background values saved to {os.path.abspath(args.output_file)}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

if __name__ == "__main__":
    import glob  
    main()
