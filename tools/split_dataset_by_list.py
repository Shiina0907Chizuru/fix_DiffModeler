import os
import shutil
import argparse
import re
from tqdm import tqdm

def read_selection_file(file_path):
    """
    读取蛋白质密度选择文件，提取蛋白质ID和文件序号
    
    Args:
        file_path: 选择文件路径
        
    Returns:
        dict: 蛋白质ID到文件序号集合的映射
        set: 没有符合条件区域的蛋白质ID集合
    """
    selections = {}
    no_selection_proteins = set()  # 存储没有符合条件区域的蛋白质
    current_protein = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和标题行
                if not line or line.startswith('密度选择') or line.startswith('处理时间') or line.startswith('总蛋白质数'):
                    continue
                
                # 检查是否是新的蛋白质标记行
                protein_match = re.match(r'蛋白质:\s*(\w+)', line)
                if protein_match:
                    current_protein = protein_match.group(1)
                    continue
                
                # 检查是否有"没有符合条件的区域"标记
                no_selection_match = re.match(r'(\w+):\s*没有符合条件的区域', line)
                if no_selection_match:
                    protein_id = no_selection_match.group(1)
                    no_selection_proteins.add(protein_id)
                    print(f"  {protein_id}: 没有符合条件的区域")
                    continue
                
                # 检查是否是蛋白质ID开头的行（包含索引列表）
                protein_indices_match = re.match(r'(\w+):\s*([\d,\s]+)', line)
                if protein_indices_match:
                    protein_id = protein_indices_match.group(1)
                    indices_str = protein_indices_match.group(2)
                    
                    # 解析索引
                    indices = [idx.strip() for idx in indices_str.split(',')]
                    
                    if protein_id not in selections:
                        selections[protein_id] = set()
                    
                    selections[protein_id].update(indices)
                
                # 如果有当前蛋白质但没有匹配到完整行，则假设是继续的索引行
                elif current_protein and re.match(r'[\d,\s]+', line):
                    indices = [idx.strip() for idx in line.split(',')]
                    
                    if current_protein not in selections:
                        selections[current_protein] = set()
                    
                    selections[current_protein].update(indices)
        
        # 打印调试信息
        print(f"从选择文件中解析出 {len(selections)} 个有选择的蛋白质:")
        for protein_id, indices in selections.items():
            print(f"  {protein_id}: {len(indices)} 个文件 ({', '.join(list(indices)[:5])}{'...' if len(indices) > 5 else ''})")
        
        print(f"从选择文件中解析出 {len(no_selection_proteins)} 个没有符合条件区域的蛋白质")
    
    except Exception as e:
        print(f"错误: 无法解析选择文件 {file_path}: {str(e)}")
        return {}, set()
    
    return selections, no_selection_proteins

def process_dataset(base_dataset, selection_file, selected_output, remaining_output, verbose=False):
    """
    处理数据集，将选定的文件和剩余文件分别复制到不同的目录
    
    Args:
        base_dataset: 基准数据集路径
        selection_file: 选择文件路径
        selected_output: 选定文件的输出目录
        remaining_output: 剩余文件的输出目录
        verbose: 是否显示详细信息
    """
    # 读取选择文件
    selections, no_selection_proteins = read_selection_file(selection_file)
    
    if not selections and not no_selection_proteins:
        print("错误: 无法从选择文件中提取有效信息")
        return
    
    # 创建输出目录
    os.makedirs(selected_output, exist_ok=True)
    os.makedirs(remaining_output, exist_ok=True)
    
    # 统计变量
    total_proteins = 0
    total_selected_pairs = 0
    total_remaining_pairs = 0
    processed_proteins = []
    
    # 处理基准数据集中的每个蛋白质文件夹
    protein_folders = [f for f in os.listdir(base_dataset) if os.path.isdir(os.path.join(base_dataset, f))]
    
    for protein_id in tqdm(protein_folders, desc="处理蛋白质"):
        total_proteins += 1
        protein_folder = os.path.join(base_dataset, protein_id)
        
        # 创建对应的输出目录
        selected_protein_dir = os.path.join(selected_output, protein_id)
        remaining_protein_dir = os.path.join(remaining_output, protein_id)
        
        # 处理没有符合条件区域的蛋白质（将所有文件放入剩余目录）
        if protein_id in no_selection_proteins:
            if verbose:
                print(f"蛋白质 {protein_id} 没有符合条件的区域，所有文件将放入剩余目录")
            
            # 获取所有input文件
            input_files = [f for f in os.listdir(protein_folder) if f.startswith('input_') and f.endswith('.npy')]
            
            # 如果没有input文件，继续下一个蛋白质
            if not input_files:
                if verbose:
                    print(f"警告: 蛋白质 {protein_id} 没有找到任何input文件")
                continue
                
            processed_proteins.append(protein_id)
            os.makedirs(remaining_protein_dir, exist_ok=True)
            
            # 将所有文件复制到剩余目录
            protein_remaining_count = 0
            
            for input_file in input_files:
                # 提取文件索引
                match = re.search(r'input_(\d+)\.npy', input_file)
                if not match:
                    if verbose:
                        print(f"警告: 无法从文件名提取索引 - {input_file}")
                    continue
                    
                file_index = match.group(1)
                
                # 构建输出文件名
                output_file = f"output_{file_index}.npy"
                
                # 完整文件路径
                input_path = os.path.join(protein_folder, input_file)
                output_path = os.path.join(protein_folder, output_file)
                
                # 如果output文件不存在，跳过此文件对
                if not os.path.exists(output_path):
                    if verbose:
                        print(f"警告: 输出文件不存在 - {output_path}")
                    continue
                
                # 复制到剩余目录
                shutil.copy2(input_path, remaining_protein_dir)
                shutil.copy2(output_path, remaining_protein_dir)
                protein_remaining_count += 1
                total_remaining_pairs += 1
                
                if verbose:
                    print(f"已复制剩余文件对 {protein_id}/{file_index} 到剩余目录")
            
            if verbose:
                print(f"蛋白质 {protein_id}: 选中 0 对, 剩余 {protein_remaining_count} 对")
            
            continue
        
        # 获取当前蛋白质的选择信息
        selected_indices = selections.get(protein_id, set())
        
        # 如果蛋白质不在选择列表中，视为全部保留在remaining目录
        if protein_id not in selections and protein_id not in no_selection_proteins and verbose:
            print(f"警告: 蛋白质 {protein_id} 不在选择文件中，将其所有文件视为剩余")
        
        # 获取所有input文件
        input_files = [f for f in os.listdir(protein_folder) if f.startswith('input_') and f.endswith('.npy')]
        
        # 如果没有input文件，继续下一个蛋白质
        if not input_files:
            if verbose:
                print(f"警告: 蛋白质 {protein_id} 没有找到任何input文件")
            continue
            
        processed_proteins.append(protein_id)
        
        # 处理每个文件对
        protein_selected_count = 0
        protein_remaining_count = 0
        
        for input_file in input_files:
            # 提取文件索引
            match = re.search(r'input_(\d+)\.npy', input_file)
            if not match:
                if verbose:
                    print(f"警告: 无法从文件名提取索引 - {input_file}")
                continue
                
            file_index = match.group(1)
            
            # 构建输出文件名
            output_file = f"output_{file_index}.npy"
            
            # 完整文件路径
            input_path = os.path.join(protein_folder, input_file)
            output_path = os.path.join(protein_folder, output_file)
            
            # 如果output文件不存在，跳过此文件对
            if not os.path.exists(output_path):
                if verbose:
                    print(f"警告: 输出文件不存在 - {output_path}")
                continue
            
            # 确定此文件对是否被选中
            is_selected = file_index in selected_indices
            
            # 确保目录存在
            if is_selected and not os.path.exists(selected_protein_dir):
                os.makedirs(selected_protein_dir, exist_ok=True)
            
            if not is_selected and not os.path.exists(remaining_protein_dir):
                os.makedirs(remaining_protein_dir, exist_ok=True)
            
            # 复制文件到相应目录
            if is_selected:
                total_selected_pairs += 1
                protein_selected_count += 1
                # 复制到选定目录
                shutil.copy2(input_path, selected_protein_dir)
                shutil.copy2(output_path, selected_protein_dir)
                if verbose:
                    print(f"已复制选定文件对 {protein_id}/{file_index} 到选定目录")
            else:
                total_remaining_pairs += 1
                protein_remaining_count += 1
                # 复制到剩余目录
                shutil.copy2(input_path, remaining_protein_dir)
                shutil.copy2(output_path, remaining_protein_dir)
                if verbose:
                    print(f"已复制剩余文件对 {protein_id}/{file_index} 到剩余目录")
        
        # 如果一个蛋白质的所有文件都被归类完成，并且没有选中任何文件，则移除其选定目录
        if protein_selected_count == 0 and os.path.exists(selected_protein_dir):
            shutil.rmtree(selected_protein_dir)
            if verbose:
                print(f"移除空目录: {selected_protein_dir}")
        
        # 如果一个蛋白质的所有文件都被选中，则移除其剩余目录
        if protein_remaining_count == 0 and os.path.exists(remaining_protein_dir):
            shutil.rmtree(remaining_protein_dir)
            if verbose:
                print(f"移除空目录: {remaining_protein_dir}")
        
        if verbose:
            print(f"蛋白质 {protein_id}: 选中 {protein_selected_count} 对, 剩余 {protein_remaining_count} 对")
    
    # 输出统计信息
    print("\n处理完成!")
    print(f"总共处理了 {len(processed_proteins)}/{total_proteins} 个蛋白质")
    print(f"复制了 {total_selected_pairs} 对选定文件到 {selected_output}")
    print(f"复制了 {total_remaining_pairs} 对剩余文件到 {remaining_output}")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='根据选择文件将数据集分割为选定和剩余两部分')
    parser.add_argument('--base_dataset', type=str, required=True,
                        help='基准数据集路径')
    parser.add_argument('--selection_file', type=str, required=True,
                        help='选择文件路径，包含要选择的蛋白质和文件序号')
    parser.add_argument('--selected_output', type=str, required=True,
                        help='选定文件的输出目录')
    parser.add_argument('--remaining_output', type=str, required=True,
                        help='剩余文件的输出目录')
    parser.add_argument('--verbose', action='store_true',
                        help='是否显示详细信息')
    
    args = parser.parse_args()
    
    # 处理数据集
    process_dataset(
        base_dataset=args.base_dataset,
        selection_file=args.selection_file,
        selected_output=args.selected_output,
        remaining_output=args.remaining_output,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

# 示例用法:
# python split_dataset_by_list.py --base_dataset "/defaultShare/zcan-library/Diffmodeler_data/20250310processed_dataset" --selection_file "C:\Users\Z\Desktop\all_proteins_density_select.txt" --selected_output "C:\Users\Z\Desktop\selected_dataset" --remaining_output "C:\Users\Z\Desktop\remaining_dataset" --verbose
