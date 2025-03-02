import os
import shutil
import glob

def process_dataset(source_root, target_root):
    """
    处理数据集，将每个蛋白质的 input 和 output npy 文件移动到新的目录结构中
    
    Args:
        source_root: 源数据集根目录，包含多个蛋白质文件夹
        target_root: 目标数据集根目录，将按照新的目录结构组织数据
    """
    # 确保目标目录存在
    os.makedirs(target_root, exist_ok=True)
    
    # 用于存储文件数量不匹配的蛋白质
    mismatched_proteins = []
    
    # 遍历源目录中的所有蛋白质文件夹
    for protein_dir in os.listdir(source_root):
        protein_path = os.path.join(source_root, protein_dir)
        if not os.path.isdir(protein_path):
            continue
            
        # 提取蛋白质ID（保持原始格式，如3j22）
        protein_id = protein_dir.replace('PDB-', '').split('-')[0].lower()  # 转换为小写以保持一致性
        print(f"Processing protein: {protein_id}")
        
        # 构建数据集路径
        dataset_path = os.path.join(protein_path, 'backbone_Dataset')
        if not os.path.exists(dataset_path):
            print(f"Dataset directory not found for {protein_id}")
            continue
            
        # 创建目标蛋白质目录（使用简单的ID命名）
        target_protein_dir = os.path.join(target_root, protein_id)
        os.makedirs(target_protein_dir, exist_ok=True)
        
        # 查找并移动input文件
        input_files = glob.glob(os.path.join(dataset_path, '*input*.npy'))
        output_files = glob.glob(os.path.join(dataset_path, '*output*.npy'))
        
        if not input_files or not output_files:
            print(f"No input/output files found for {protein_id}")
            continue
            
        print(f"Found {len(input_files)} input files and {len(output_files)} output files")
        
        # 检查文件数量是否匹配
        if len(input_files) != len(output_files):
            print(f"Warning: Number of input files ({len(input_files)}) does not match number of output files ({len(output_files)}) for {protein_id}")
            mismatched_proteins.append({
                'protein_id': protein_id,
                'input_count': len(input_files),
                'output_count': len(output_files)
            })
        
        # 移动文件
        for input_file in input_files:
            input_filename = os.path.basename(input_file)
            target_input = os.path.join(target_protein_dir, input_filename)
            shutil.move(input_file, target_input)
            print(f"Moved {input_filename}")
            
        for output_file in output_files:
            output_filename = os.path.basename(output_file)
            target_output = os.path.join(target_protein_dir, output_filename)
            shutil.move(output_file, target_output)
            print(f"Moved {output_filename}")
            
        print(f"Completed processing {protein_id}")

    # 打印文件数量不匹配的蛋白质信息
    if mismatched_proteins:
        print("\n=== Proteins with mismatched file counts ===")
        print("Protein ID\tInput Files\tOutput Files")
        print("-" * 50)
        for protein in mismatched_proteins:
            print(f"{protein['protein_id']}\t{protein['input_count']}\t{protein['output_count']}")
    else:
        print("\nAll proteins have matching input and output file counts.")

if __name__ == "__main__":
    # 源数据集路径
    source_root = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\newdateset\trainpdb_emdb_data"
    # 目标数据集路径
    target_root = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\newdateset\dataset"
    
    # 处理数据集
    process_dataset(source_root, target_root)
    print("Dataset processing completed!")
