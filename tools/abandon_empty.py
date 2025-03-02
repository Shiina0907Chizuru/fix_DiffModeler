import os
import numpy as np
import shutil
from tqdm import tqdm

def is_empty_output(file_path):
    """检查output文件是否为空（全0）
    Args:
        file_path: output.npy文件路径
    Returns:
        bool: 是否为空
    """
    try:
        data = np.load(file_path)
        return np.all(data == 0)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path} - {str(e)}")
        return True

def process_protein_folder(protein_folder, backup_dir=None):
    """处理单个蛋白质文件夹
    Args:
        protein_folder: 蛋白质文件夹路径
        backup_dir: 备份目录，如果指定则会备份被删除的文件
    Returns:
        tuple: (处理的文件对数量, 删除的文件对数量)
    """
    total_pairs = 0
    empty_pairs = 0
    
    # 获取所有output文件
    output_files = [f for f in os.listdir(protein_folder) if f.startswith('output_') and f.endswith('.npy')]
    
    for output_file in output_files:
        total_pairs += 1
        output_path = os.path.join(protein_folder, output_file)
        input_file = 'input_' + output_file.split('_')[1]
        input_path = os.path.join(protein_folder, input_file)
        
        if is_empty_output(output_path):
            empty_pairs += 1
            
            # 如果指定了备份目录，先备份文件
            if backup_dir:
                protein_name = os.path.basename(protein_folder)
                backup_protein_dir = os.path.join(backup_dir, protein_name)
                os.makedirs(backup_protein_dir, exist_ok=True)
                
                if os.path.exists(input_path):
                    shutil.copy2(input_path, backup_protein_dir)
                if os.path.exists(output_path):
                    shutil.copy2(output_path, backup_protein_dir)
            
            # 删除文件
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    
    return total_pairs, empty_pairs

def main():
    # 配置
    dataset_root = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/dataset"
    protein_list_file = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/pdb_ids.txt"  # 包含蛋白质名称的文件
    backup_dir = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/empty_backup"  # 备份目录
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 读取蛋白质列表
    with open(protein_list_file, 'r') as f:
        proteins = [line.strip() for line in f.readlines()]
    
    # 统计信息
    total_processed = 0
    total_empty = 0
    protein_stats = []
    
    # 处理每个蛋白质文件夹
    print("开始处理数据...")
    for protein in tqdm(proteins):
        protein_folder = os.path.join(dataset_root, protein)
        if not os.path.exists(protein_folder):
            print(f"警告: 文件夹不存在 - {protein_folder}")
            continue
            
        pairs, empty = process_protein_folder(protein_folder, backup_dir)
        total_processed += pairs
        total_empty += empty
        
        if empty > 0:
            protein_stats.append(f"{protein}: {empty}/{pairs} 个文件对被删除")
    
    # 打印统计信息
    print("\n处理完成!")
    print(f"总共处理: {total_processed} 个文件对")
    print(f"删除空文件: {total_empty} 个文件对")
    print("\n各蛋白质处理统计:")
    for stat in protein_stats:
        print(stat)
    
    # 保存处理日志
    log_file = "abandon_empty_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"总共处理: {total_processed} 个文件对\n")
        f.write(f"删除空文件: {total_empty} 个文件对\n\n")
        f.write("各蛋白质处理统计:\n")
        for stat in protein_stats:
            f.write(stat + '\n')
    
    print(f"\n处理日志已保存到: {log_file}")

if __name__ == "__main__":
    main()
