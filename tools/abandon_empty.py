import os
import numpy as np
import shutil
from tqdm import tqdm
import argparse
from datetime import datetime

def load_background_info(background_file):
    """从背景信息文件加载蛋白质特定的阈值信息
    
    Args:
        background_file: 背景信息文件路径
        
    Returns:
        dict: 蛋白质ID到阈值信息的映射，格式为:
            {
                'protein_id': {
                    'inputbackground': float,
                    'outputbackground': float
                }
            }
    注意: 虽然背景文件中可能包含百分比信息，但本函数只会使用background阈值
    """
    background_data = {}
    
    try:
        with open(background_file, 'r') as f:
            # 跳过注释行
            for line in f:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 9:  # 确保有足够的字段
                    protein_id = parts[0]
                    
                    # 提取数值，处理前缀
                    input_bg = float(parts[1].split(':')[1]) if ':' in parts[1] else float(parts[2])
                    output_bg = float(parts[5].split(':')[1]) if ':' in parts[5] else float(parts[6])
                    
                    background_data[protein_id] = {
                        'inputbackground': input_bg,
                        'outputbackground': output_bg
                    }
        
        print(f"成功从 {background_file} 加载了 {len(background_data)} 个蛋白质的背景信息")
    except Exception as e:
        print(f"警告: 无法加载背景信息文件 {background_file}: {str(e)}")
    
    return background_data

def is_empty_data(file_path, threshold=0, min_percent=5.0):
    """检查文件是否为空（不足指定百分比的数据超过阈值）
    
    Args:
        file_path: .npy文件路径
        threshold: 判断文件为空的阈值
        min_percent: 最低要求的超过阈值的数据百分比
        
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

def process_protein_folder(protein_folder, background_data=None, filter_mode="output", 
                          backup_dir=None, output_dir=None, verbose=False, 
                          default_threshold=0, default_min_percent=5.0,
                          operation_type="cut", cluster_mode=False):
    """处理单个蛋白质文件夹
    
    Args:
        protein_folder: 蛋白质文件夹路径
        background_data: 背景信息数据，用于获取特定蛋白质的阈值
        filter_mode: 过滤模式，"input"表示基于input文件过滤，"output"表示基于output文件过滤
        backup_dir: 备份目录，用于存放空文件或不满足条件的文件
        output_dir: 输出目录，仅在cluster_mode=True时使用，用于存放满足条件的文件
        verbose: 是否打印详细信息
        default_threshold: 默认判断文件为空的阈值（当背景文件中没有对应蛋白质时使用）
        default_min_percent: 默认最低要求的超过阈值的数据百分比（当背景文件中没有对应蛋白质时使用）
        operation_type: 操作类型，"cut"表示剪切（删除原文件），"copy"表示复制（保留原文件）
        cluster_mode: 是否为集群处理模式
        
    Returns:
        tuple: (处理的文件对数量, 操作的文件对数量)
    """
    total_pairs = 0
    empty_pairs = 0  # 初始化空文件对计数
    
    # 获取蛋白质名称
    protein_name = os.path.basename(protein_folder)
    
    # 获取特定蛋白质的阈值设置
    threshold = default_threshold
    min_percent = default_min_percent  # 最小百分比始终使用命令行参数指定的值
    
    if background_data and protein_name in background_data:
        if filter_mode == "input":
            threshold = background_data[protein_name]['inputbackground']
        else:  # filter_mode == "output"
            threshold = background_data[protein_name]['outputbackground']
            
        if verbose:
            print(f"使用蛋白质 {protein_name} 的特定阈值: threshold={threshold}, min_percent={min_percent}")
    elif verbose:
        print(f"使用默认阈值: threshold={threshold}, min_percent={min_percent}")
    
    # 获取所有文件对（根据过滤模式）
    if filter_mode == "input":
        # 基于input文件过滤
        input_files = [f for f in os.listdir(protein_folder) if f.startswith('input_') and f.endswith('.npy')]
        if verbose:
            print(f"处理文件夹: {protein_folder}, 找到 {len(input_files)} 个input文件")
        
        for input_file in input_files:
            total_pairs += 1
            
            # 构造对应的output文件名
            output_file = input_file.replace('input_', 'output_')
            
            # 文件完整路径
            input_path = os.path.join(protein_folder, input_file)
            output_path = os.path.join(protein_folder, output_file)
            
            # 检查文件是否为空
            is_empty = is_empty_data(input_path, threshold, min_percent)
            
            # 处理空文件对
            if is_empty:
                empty_pairs += 1
                process_file_pair(input_file, output_file, input_path, output_path, 
                                 protein_name, protein_folder, backup_dir, output_dir,
                                 verbose, operation_type, cluster_mode, is_empty=True)
            elif cluster_mode and output_dir:
                # 处理非空（有效）文件对
                process_file_pair(input_file, output_file, input_path, output_path, 
                                 protein_name, protein_folder, backup_dir, output_dir,
                                 verbose, operation_type, cluster_mode, is_empty=False)
    else:  # filter_mode == "output"
        # 基于output文件过滤（原有功能）
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
            is_empty = is_empty_data(output_path, threshold, min_percent)
            
            # 处理空文件对
            if is_empty:
                empty_pairs += 1
                process_file_pair(input_file, output_file, input_path, output_path, 
                                 protein_name, protein_folder, backup_dir, output_dir,
                                 verbose, operation_type, cluster_mode, is_empty=True)
            elif cluster_mode and output_dir:
                # 处理非空（有效）文件对
                process_file_pair(input_file, output_file, input_path, output_path, 
                                 protein_name, protein_folder, backup_dir, output_dir,
                                 verbose, operation_type, cluster_mode, is_empty=False)
    
    if verbose:
        print(f"文件夹 {protein_folder} 处理完成: 总计 {total_pairs} 对文件")
        if cluster_mode:
            print(f"  {empty_pairs} 对空文件被复制到备份目录")
            print(f"  {total_pairs - empty_pairs} 对有效文件被复制到输出目录")
        else:
            operation_msg = "删除" if operation_type == "cut" else "复制"
            print(f"  {empty_pairs} 对空文件被{operation_msg}")
        
    return total_pairs, empty_pairs

def process_file_pair(input_file, output_file, input_path, output_path, 
                     protein_name, protein_folder, backup_dir, output_dir,
                     verbose, operation_type, cluster_mode, is_empty):
    """处理单个文件对
    
    Args:
        input_file: 输入文件名
        output_file: 输出文件名
        input_path: 输入文件完整路径
        output_path: 输出文件完整路径
        protein_name: 蛋白质名称
        protein_folder: 蛋白质文件夹路径
        backup_dir: 备份目录
        output_dir: 输出目录（仅在cluster_mode=True时使用）
        verbose: 是否打印详细信息
        operation_type: 操作类型（"cut"或"copy"）
        cluster_mode: 是否为集群处理模式
        is_empty: 是否为空文件对
    """
    # 获取操作消息
    if cluster_mode:
        operation_msg = "移除" if is_empty else "保留"
    else:
        operation_msg = "删除" if operation_type == "cut" else "复制"
    
    if is_empty:
        if verbose:
            print(f"发现空文件对: input_{input_file.split('input_')[1]} / output_{output_file.split('output_')[1]}")
        
        # 创建备份目录（如果需要）
        if backup_dir:
            backup_protein_dir = os.path.join(backup_dir, protein_name)
            os.makedirs(backup_protein_dir, exist_ok=True)
            
            # 处理输入文件
            if os.path.exists(input_path):
                shutil.copy2(input_path, backup_protein_dir)
                if verbose:
                    print(f"  备份文件: {input_file} -> {backup_protein_dir}")
                    
                # 只有在剪切模式且不是集群模式时才删除源文件
                if operation_type == "cut" and not cluster_mode:
                    os.remove(input_path)
                    if verbose:
                        print(f"  删除文件: {input_file}")
            
            # 处理输出文件
            if os.path.exists(output_path):
                shutil.copy2(output_path, backup_protein_dir)
                if verbose:
                    print(f"  备份文件: {output_file} -> {backup_protein_dir}")
                    
                # 只有在剪切模式且不是集群模式时才删除源文件
                if operation_type == "cut" and not cluster_mode:
                    os.remove(output_path)
                    if verbose:
                        print(f"  删除文件: {output_file}")
        
        # 如果没有备份目录但仍需要删除文件（本地剪切模式）
        elif operation_type == "cut" and not cluster_mode:
            if os.path.exists(input_path):
                os.remove(input_path)
                if verbose:
                    print(f"  删除文件: {input_file}")
                    
            if os.path.exists(output_path):
                os.remove(output_path)
                if verbose:
                    print(f"  删除文件: {output_file}")
    
    # 处理非空文件（仅在集群模式且有输出目录时）
    elif cluster_mode and output_dir:
        output_protein_dir = os.path.join(output_dir, protein_name)
        os.makedirs(output_protein_dir, exist_ok=True)
        
        # 复制输入文件到输出目录
        if os.path.exists(input_path):
            shutil.copy2(input_path, output_protein_dir)
            if verbose:
                print(f"  复制有效文件到输出目录: {input_file} -> {output_protein_dir}")
        
        # 复制输出文件到输出目录
        if os.path.exists(output_path):
            shutil.copy2(output_path, output_protein_dir)
            if verbose:
                print(f"  复制有效文件到输出目录: {output_file} -> {output_protein_dir}")

def adaptive_process_protein_folder(protein_folder, background_data=None, filter_mode="output", 
                                   backup_dir=None, output_dir=None, verbose=False, 
                                   default_threshold=0, default_min_percent=5.0,
                                   operation_type="cut", cluster_mode=False,
                                   adaptive_increment=0.00001, adaptive_max_iterations=100,
                                   target_min_percent=5.0, target_proteins=None):
    """使用自适应阈值处理单个蛋白质文件夹
    
    与process_protein_folder函数相比，该函数增加了自适应调整阈值的功能，
    确保空文件比例达到预期目标
    
    Args:
        protein_folder: 蛋白质文件夹路径
        background_data: 背景信息数据，用于获取特定蛋白质的阈值
        filter_mode: 过滤模式，"input"表示基于input文件过滤，"output"表示基于output文件过滤
        backup_dir: 备份目录，用于存放空文件或不满足条件的文件
        output_dir: 输出目录，仅在cluster_mode=True时使用，用于存放满足条件的文件
        verbose: 是否打印详细信息
        default_threshold: 默认判断文件为空的阈值（当背景文件中没有对应蛋白质时使用）
        default_min_percent: 默认最低要求的超过阈值的数据百分比（当背景文件中没有对应蛋白质时使用）
        operation_type: 操作类型，"cut"表示剪切（删除原文件），"copy"表示复制（保留原文件）
        cluster_mode: 是否为集群处理模式
        adaptive_increment: 每次迭代增加的阈值
        adaptive_max_iterations: 最大迭代次数
        target_min_percent: 目标最小空文件比例
        target_proteins: 需要优先处理的目标蛋白质列表
        
    Returns:
        tuple: (处理的文件对数量, 操作的文件对数量, 最终使用的阈值)
    """
    protein_name = os.path.basename(protein_folder)
    
    # 检查是否为目标蛋白质
    is_target = target_proteins is not None and protein_name in target_proteins
    
    # 如果不是目标蛋白质但提供了目标蛋白质列表，则使用普通处理方式
    if target_proteins is not None and not is_target:
        return process_protein_folder(
            protein_folder=protein_folder,
            background_data=background_data,
            filter_mode=filter_mode,
            backup_dir=backup_dir,
            output_dir=output_dir,
            verbose=verbose,
            default_threshold=default_threshold,
            default_min_percent=default_min_percent,
            operation_type=operation_type,
            cluster_mode=cluster_mode
        ) + (default_threshold,)  # 添加使用的阈值作为返回值的第三个元素
    
    # 获取蛋白质名称和初始阈值设置
    threshold = default_threshold
    min_percent = default_min_percent
    
    if background_data and protein_name in background_data:
        if filter_mode == "input":
            threshold = background_data[protein_name]['inputbackground']
        else:  # filter_mode == "output"
            threshold = background_data[protein_name]['outputbackground']
    
    # 初始阈值
    current_threshold = threshold
    current_iteration = 0
    
    # 收集所有文件路径用于评估
    if filter_mode == "input":
        files = [f for f in os.listdir(protein_folder) if f.startswith('input_') and f.endswith('.npy')]
        file_paths = [os.path.join(protein_folder, f) for f in files]
    else:  # filter_mode == "output"
        files = [f for f in os.listdir(protein_folder) if f.startswith('output_') and f.endswith('.npy')]
        file_paths = [os.path.join(protein_folder, f) for f in files]
    
    total_files = len(file_paths)
    
    if total_files == 0:
        if verbose:
            print(f"警告: 文件夹 {protein_folder} 没有找到任何{filter_mode}文件")
        return 0, 0, threshold
    
    if verbose:
        print(f"开始对蛋白质 {protein_name} 进行自适应阈值处理，初始阈值: {current_threshold}")
    
    # 首先检查在当前阈值下是否有文件需要被移动
    empty_count = sum(1 for path in file_paths if is_empty_data(path, current_threshold, min_percent))
    deletion_percent = (empty_count / total_files) * 100
    
    if verbose:
        if empty_count == 0:
            print(f"  在初始阈值 {current_threshold} 下没有需要移动的文件，开始调整阈值")
        else:
            print(f"  迭代 1: 阈值 = {current_threshold:.6f}, 空文件比例 = {deletion_percent:.2f}% ({empty_count}/{total_files})")
    
    # 如果当前没有文件需要移动或者比例不足，则逐步增加阈值
    while (empty_count == 0 or deletion_percent < target_min_percent) and current_iteration < adaptive_max_iterations:
        # 增加阈值
        current_threshold += adaptive_increment
        current_iteration += 1
        
        # 重新计算在新阈值下的空文件数量
        empty_count = sum(1 for path in file_paths if is_empty_data(path, current_threshold, min_percent))
        deletion_percent = (empty_count / total_files) * 100
        
        if verbose:
            print(f"  迭代 {current_iteration+1}: 阈值 = {current_threshold:.6f}, 空文件比例 = {deletion_percent:.2f}% ({empty_count}/{total_files})")
        
        # 检查是否达到目标
        if deletion_percent >= target_min_percent:
            if verbose:
                print(f"  已达到目标空文件比例 {target_min_percent}%，停止迭代")
            break
    
    if current_iteration == adaptive_max_iterations and verbose:
        print(f"  警告: 达到最大迭代次数 ({adaptive_max_iterations})，但未达到目标空文件比例")
    elif empty_count > 0 and deletion_percent >= target_min_percent and verbose:
        print(f"  已有 {empty_count} 个空文件({deletion_percent:.2f}%)，满足最低要求{target_min_percent}%，无需调整")
    
    # 使用最终确定的阈值处理文件
    if verbose:
        print(f"使用最终阈值 {current_threshold:.6f} 处理蛋白质 {protein_name}")
    
    pairs, empty = process_protein_folder(
        protein_folder=protein_folder,
        background_data=None,  # 不使用背景文件中的阈值，而是使用自适应调整后的阈值
        filter_mode=filter_mode,
        backup_dir=backup_dir,
        output_dir=output_dir,
        verbose=verbose,
        default_threshold=current_threshold,  # 使用调整后的阈值
        default_min_percent=min_percent,
        operation_type=operation_type,
        cluster_mode=cluster_mode
    )
    
    return pairs, empty, current_threshold

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='检测并处理空的数据文件对')
    parser.add_argument('--dataset_root', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/dataset",
                        help='数据集根目录')
    parser.add_argument('--protein_list', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/pdb_ids.txt",
                        help='蛋白质ID列表文件')
    parser.add_argument('--background_file', type=str,
                        default="",
                        help='背景信息文件，包含每个蛋白质的阈值信息')
    parser.add_argument('--filter_mode', type=str, choices=['input', 'output'], default='output',
                        help='过滤模式，input表示根据input文件过滤，output表示根据output文件过滤')
    parser.add_argument('--backup_dir', type=str, 
                        default="E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/empty_backup",
                        help='备份目录，用于存放空文件，如果不需要备份可设为空字符串')
    parser.add_argument('--output_dir', type=str, default="", 
                        help='输出目录，仅在集群模式下使用，用于存放有效文件')
    parser.add_argument('--log_file', type=str, default="abandon_empty_log.txt", help='日志文件名')
    parser.add_argument('--threshold', type=float, default=0, 
                        help='默认判断文件为空的阈值，大于此值的数据才被视为有效数据（当背景文件中没有对应蛋白质时使用）')
    parser.add_argument('--min_percent', type=float, default=10.0,
                        help='最低要求的超过阈值的数据百分比，适用于所有蛋白质')
    parser.add_argument('--verbose', action='store_true', help='是否显示详细信息')
    parser.add_argument('--operation', type=str, choices=['cut', 'copy'], default='cut',
                        help='操作类型，cut表示剪切（删除原文件），copy表示复制（保留原文件）')
    parser.add_argument('--cluster_mode', action='store_true', 
                        help='是否使用集群处理模式，该模式下原数据集不变，有效文件被复制到输出目录')
    parser.add_argument('--adaptive_increment', type=float, default=0.00001, 
                        help='自适应阈值调整的增量')
    parser.add_argument('--adaptive_max_iterations', type=int, default=100, 
                        help='自适应阈值调整的最大迭代次数')
    parser.add_argument('--target_min_percent', type=float, default=10.0, 
                        help='目标最小空文件比例')
    parser.add_argument('--target_proteins', type=str, default=None, 
                        help='需要优先处理的目标蛋白质列表文件路径')
    
    args = parser.parse_args()
    
    # 配置
    dataset_root = args.dataset_root
    protein_list_file = args.protein_list
    background_file = args.background_file
    filter_mode = args.filter_mode
    backup_dir = args.backup_dir if args.backup_dir else None
    output_dir = args.output_dir if args.output_dir and args.cluster_mode else None
    log_file = args.log_file
    verbose = args.verbose
    operation_type = args.operation
    cluster_mode = args.cluster_mode
    adaptive_increment = args.adaptive_increment
    adaptive_max_iterations = args.adaptive_max_iterations
    target_min_percent = args.target_min_percent
    
    # 加载背景信息文件（如果提供）
    background_data = None
    if background_file:
        background_data = load_background_info(background_file)
    
    # 加载目标蛋白质列表（如果提供）
    target_proteins = None
    if args.target_proteins:
        try:
            with open(args.target_proteins, 'r') as f:
                target_proteins = [line.strip() for line in f.readlines()]
            print(f"从 {args.target_proteins} 读取了 {len(target_proteins)} 个目标蛋白质")
        except Exception as e:
            print(f"错误: 无法读取目标蛋白质列表文件 ({args.target_proteins}): {str(e)}")
    
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
    
    # 打印处理模式
    mode_msg = "集群处理" if cluster_mode else f"本地处理({operation_type}模式)"
    filter_msg = f"{filter_mode}过滤模式"
    threshold_msg = "使用蛋白质特定阈值" if background_data else f"使用默认阈值(threshold={args.threshold}, min_percent={args.min_percent}%)"
    
    print(f"开始处理 {dataset_root} 数据，使用{mode_msg}，{filter_msg}，{threshold_msg}...")
    
    # 处理每个蛋白质文件夹
    for protein in tqdm(proteins):
        protein_folder = os.path.join(dataset_root, protein)
        if not os.path.exists(protein_folder):
            print(f"警告: 文件夹不存在 - {protein_folder}")
            continue
            
        pairs, empty, threshold = adaptive_process_protein_folder(
            protein_folder=protein_folder,
            background_data=background_data,
            filter_mode=filter_mode,
            backup_dir=backup_dir, 
            output_dir=output_dir,
            verbose=verbose, 
            default_threshold=args.threshold, 
            default_min_percent=args.min_percent,
            operation_type=operation_type,
            cluster_mode=cluster_mode,
            adaptive_increment=adaptive_increment,
            adaptive_max_iterations=adaptive_max_iterations,
            target_min_percent=target_min_percent,
            target_proteins=target_proteins
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
        f.write(f"过滤模式: {filter_mode}\n")
        f.write(f"背景信息文件: {background_file if background_file else '无'}\n")
        f.write(f"处理模式: {'集群处理' if cluster_mode else '本地处理'}\n")
        if cluster_mode:
            f.write(f"输出目录: {output_dir if output_dir else '无'}\n")
        f.write(f"备份目录: {backup_dir if backup_dir else '无'}\n")
        
        if not background_data:
            f.write(f"默认阈值设置: {args.threshold}\n")
            f.write(f"默认最低百分比设置: {args.min_percent}%\n")
        else:
            f.write(f"使用蛋白质特定阈值\n")
            
        if not cluster_mode:
            f.write(f"操作类型: {operation_type} ({operation_msg})\n")
        f.write("\n")
        
        f.write(f"总共处理: {total_processed} 个文件对\n")
        if cluster_mode:
            f.write(f"空文件（已复制到备份目录）: {total_empty} 个文件对\n")
            f.write(f"有效文件（已复制到输出目录）: {total_processed - total_empty} 个文件对\n")
        else:
            f.write(f"{operation_msg}空文件: {total_empty} 个文件对\n")
            
        if total_processed > 0:
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