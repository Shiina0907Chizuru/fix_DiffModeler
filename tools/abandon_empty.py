import os
import numpy as np
import shutil
from tqdm import tqdm
import argparse
from datetime import datetime

def is_empty_output(file_path, threshold=0, min_percent=5.0):
    """检查output文件是否为空（不足指定百分比的数据超过阈值）
    Args:
        file_path: output.npy文件路径
        threshold: 判断文件为空的阈值
        min_percent: 最低要求的超过阈值的数据百分比，默认为5%
    Returns:
        bool: 是否为空（返回True表示为空）
    """
    try:
        data = np.load(file_path)
        
        # 计算超过阈值的数据点百分比
        above_threshold = np.sum(data > threshold)
        total_points = data.size
        percent_above = (above_threshold / total_points) * 100
        
        # 如果低于最低要求百分比，则视为空文件
        return percent_above < min_percent
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path} - {str(e)}")
        return True

def process_protein_folder(protein_folder, backup_dir=None, output_dir=None, verbose=False, threshold=0, min_percent=5.0, operation_type="cut", cluster_mode=False):
    """处理单个蛋白质文件夹
    Args:
        protein_folder: 蛋白质文件夹路径
        backup_dir: 备份目录，用于存放空文件或不满足条件的文件
        output_dir: 输出目录，仅在cluster_mode=True时使用，用于存放满足条件的文件
        verbose: 是否打印详细信息
        threshold: 判断文件为空的阈值
        min_percent: 最低要求的超过阈值的数据百分比，默认为5%
        operation_type: 操作类型，"cut"表示剪切（删除原文件），"copy"表示复制（保留原文件）
        cluster_mode: 是否为集群处理模式
    Returns:
        tuple: (处理的文件对数量, 操作的文件对数量)
    """
    total_pairs = 0
    empty_pairs = 0
    
    # 获取蛋白质名称
    protein_name = os.path.basename(protein_folder)
    
    # 获取所有output文件
    output_files = [f for f in os.listdir(protein_folder) if f.startswith('output_') and f.endswith('.npy')]
    
    if verbose:
        print(f"处理文件夹: {protein_folder}, 找到 {len(output_files)} 个output文件")
    
    for output_file in output_files:
        total_pairs += 1
        
        # 构造对应的input文件名
        input_file = output_file.replace('output_', 'input_')
        
        # 文件完整路径
        output_path = os.path.join(protein_folder, output_file)
        input_path = os.path.join(protein_folder, input_file)
        
        # 检查文件是否为空
        is_empty = is_empty_output(output_path, threshold, min_percent)
        
        if is_empty:
            empty_pairs += 1
            
            if verbose:
                print(f"发现空文件: {output_file} (对应输入文件: {input_file})")
            
            # 集群处理模式
            if cluster_mode:
                # 创建备份目录结构
                if backup_dir:
                    backup_protein_dir = os.path.join(backup_dir, protein_name)
                    os.makedirs(backup_protein_dir, exist_ok=True)
                    
                    # 复制空文件到备份目录
                    if os.path.exists(input_path):
                        shutil.copy2(input_path, backup_protein_dir)
                        if verbose:
                            print(f"  复制空输入文件到备份目录: {input_file} -> {backup_protein_dir}")
                            
                    if os.path.exists(output_path):
                        shutil.copy2(output_path, backup_protein_dir)
                        if verbose:
                            print(f"  复制空输出文件到备份目录: {output_file} -> {backup_protein_dir}")
                operation_msg = "移除"
            else:
                # 本地处理模式（原有功能）
                # 如果需要备份，则先创建备份目录并复制文件
                if backup_dir:
                    backup_protein_dir = os.path.join(backup_dir, protein_name)
                    os.makedirs(backup_protein_dir, exist_ok=True)
                    
                    if os.path.exists(input_path):
                        shutil.copy2(input_path, backup_protein_dir)
                        if verbose:
                            print(f"  备份输入文件: {input_file} -> {backup_protein_dir}")
                            
                    if os.path.exists(output_path):
                        shutil.copy2(output_path, backup_protein_dir)
                        if verbose:
                            print(f"  备份输出文件: {output_file} -> {backup_protein_dir}")
                
                # 根据操作类型选择是剪切（删除原文件）还是只复制（保留原文件）
                if operation_type == "cut":
                    # 删除文件
                    if os.path.exists(input_path):
                        os.remove(input_path)
                        if verbose:
                            print(f"  删除输入文件: {input_file}")
                            
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        if verbose:
                            print(f"  删除输出文件: {output_file}")
                    operation_msg = "删除"
                else:  # operation_type == "copy"
                    operation_msg = "复制"
                    if verbose:
                        print(f"  保留原始文件，仅复制到备份目录")
        elif cluster_mode and output_dir:
            # 在集群模式下，将非空（有效）文件复制到输出目录
            output_protein_dir = os.path.join(output_dir, protein_name)
            os.makedirs(output_protein_dir, exist_ok=True)
            
            if os.path.exists(input_path):
                shutil.copy2(input_path, output_protein_dir)
                if verbose:
                    print(f"  复制有效输入文件到输出目录: {input_file} -> {output_protein_dir}")
                    
            if os.path.exists(output_path):
                shutil.copy2(output_path, output_protein_dir)
                if verbose:
                    print(f"  复制有效输出文件到输出目录: {output_file} -> {output_protein_dir}")
    
    if verbose:
        print(f"文件夹 {protein_folder} 处理完成: 总计 {total_pairs} 对文件")
        if cluster_mode:
            print(f"  {empty_pairs} 对空文件被复制到备份目录")
            print(f"  {total_pairs - empty_pairs} 对有效文件被复制到输出目录")
        else:
            print(f"  {empty_pairs} 对空文件被{operation_msg}")
        
    return total_pairs, empty_pairs

def main():
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='检测并处理空的数据文件对')
    parser.add_argument('--dataset_root', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/dataset",
                        help='数据集根目录')
    parser.add_argument('--protein_list', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/pdb_ids.txt",
                        help='蛋白质ID列表文件')
    parser.add_argument('--backup_dir', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/empty_backup",
                        help='备份目录，用于存放空文件，如果不需要备份可设为空字符串')
    parser.add_argument('--output_dir', type=str, default="", 
                        help='输出目录，仅在集群模式下使用，用于存放有效文件')
    parser.add_argument('--log_file', type=str, default="abandon_empty_log.txt", help='日志文件名')
    parser.add_argument('--threshold', type=float, default=0, 
                        help='判断文件为空的阈值，大于此值的数据才被视为有效数据')
    parser.add_argument('--min_percent', type=float, default=5.0,
                        help='最低要求的超过阈值的数据百分比，默认为5%')
    parser.add_argument('--verbose', action='store_true', help='是否显示详细信息')
    parser.add_argument('--operation', type=str, choices=['cut', 'copy'], default='cut',
                        help='操作类型，cut表示剪切（删除原文件），copy表示复制（保留原文件）')
    parser.add_argument('--cluster_mode', action='store_true', 
                        help='是否使用集群处理模式，该模式下原数据集不变，有效文件被复制到输出目录')
    
    args = parser.parse_args()
    
    # 配置
    dataset_root = args.dataset_root
    protein_list_file = args.protein_list
    backup_dir = args.backup_dir if args.backup_dir else None
    output_dir = args.output_dir if args.output_dir and args.cluster_mode else None
    log_file = args.log_file
    verbose = args.verbose
    operation_type = args.operation
    cluster_mode = args.cluster_mode
    
    # 创建备份目录
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"备份文件将保存到: {backup_dir}")
    else:
        if cluster_mode:
            print("警告: 集群模式下未指定备份目录，空文件将不会被备份")
        else:
            print("未指定备份目录，删除的文件将不会备份")
            if operation_type == "copy":
                print("警告: 复制模式下未指定备份目录，空文件将不会被复制")
    
    # 创建输出目录（集群模式）
    if cluster_mode and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"有效文件将保存到: {output_dir}")
    elif cluster_mode and not output_dir:
        print("错误: 集群模式下必须指定输出目录")
        return
    
    # 读取蛋白质列表
    try:
        with open(protein_list_file, 'r') as f:
            proteins = [line.strip() for line in f.readlines()]
        print(f"从 {protein_list_file} 读取了 {len(proteins)} 个蛋白质ID")
    except Exception as e:
        print(f"错误: 无法读取蛋白质列表文件 ({protein_list_file}): {str(e)}")
        return
    
    # 统计信息
    total_processed = 0
    total_empty = 0
    protein_stats = []
    
    if cluster_mode:
        operation_msg = "移除到备份目录"
    else:
        operation_msg = "删除" if operation_type == "cut" else "复制"
    
    # 处理每个蛋白质文件夹
    mode_msg = "集群处理" if cluster_mode else f"本地处理({operation_type}模式)"
    print(f"开始处理 {dataset_root} 数据，使用{mode_msg}...")
    for protein in tqdm(proteins):
        protein_folder = os.path.join(dataset_root, protein)
        if not os.path.exists(protein_folder):
            print(f"警告: 文件夹不存在 - {protein_folder}")
            continue
            
        pairs, empty = process_protein_folder(
            protein_folder, 
            backup_dir=backup_dir, 
            output_dir=output_dir,
            verbose=verbose, 
            threshold=args.threshold, 
            min_percent=args.min_percent,
            operation_type=operation_type,
            cluster_mode=cluster_mode
        )
        total_processed += pairs
        total_empty += empty
        
        if empty > 0:
            protein_stats.append(f"{protein}: {empty}/{pairs} 个文件对被{operation_msg}")
    
    # 打印统计信息
    print("\n处理完成!")
    print(f"总共处理: {total_processed} 个文件对")
    if cluster_mode:
        print(f"空文件（已复制到备份目录）: {total_empty} 个文件对")
        print(f"有效文件（已复制到输出目录）: {total_processed - total_empty} 个文件对")
    else:
        print(f"{operation_msg}空文件: {total_empty} 个文件对")
    
    if total_empty > 0:
        print("\n各蛋白质处理统计:")
        for stat in protein_stats:
            print(stat)
    
    # 保存处理日志
    with open(log_file, 'w') as f:
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {dataset_root}\n")
        f.write(f"蛋白质列表: {protein_list_file}\n")
        f.write(f"处理模式: {'集群处理' if cluster_mode else '本地处理'}\n")
        if cluster_mode:
            f.write(f"输出目录: {output_dir if output_dir else '无'}\n")
        f.write(f"备份目录: {backup_dir if backup_dir else '无'}\n")
        f.write(f"阈值设置: {args.threshold}\n")
        f.write(f"最低百分比设置: {args.min_percent}%\n")
        if not cluster_mode:
            f.write(f"操作类型: {operation_type} ({operation_msg})\n")
        f.write("\n")
        
        f.write(f"总共处理: {total_processed} 个文件对\n")
        if cluster_mode:
            f.write(f"空文件（已复制到备份目录）: {total_empty} 个文件对\n")
            f.write(f"有效文件（已复制到输出目录）: {total_processed - total_empty} 个文件对\n")
        else:
            f.write(f"{operation_msg}空文件: {total_empty} 个文件对\n")
        f.write(f"空文件比例: {total_empty/total_processed*100:.2f}%\n\n")
        
        if total_empty > 0:
            f.write("各蛋白质处理统计:\n")
            for stat in protein_stats:
                f.write(stat + '\n')
                
    print(f"\n处理日志已保存到: {os.path.abspath(log_file)}")

if __name__ == "__main__":
    main()
# python abandon_empty.py --dataset_root "路径/到/数据集" --backup_dir "路径/到/备份目录" --operation cut
# python abandon_empty.py --dataset_root "路径/到/数据集" --backup_dir "路径/到/备份目录" --output_dir "路径/到/输出目录" --cluster_mode