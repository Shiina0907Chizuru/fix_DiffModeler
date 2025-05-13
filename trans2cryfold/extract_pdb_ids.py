import os
import re
import sys

def extract_pdb_ids(directory_path, output_file):
    """
    从指定目录下的文件名中提取PDB ID，并写入输出文件
    
    参数:
        directory_path: 包含文件的目录路径
        output_file: 输出文件路径
    """
    # 存储提取的PDB ID
    pdb_ids = set()
    
    # 正则表达式模式匹配文件名中的PDB ID
    # 假设文件名格式为：{pdb_id}_point_{pdb_id}_mrc2cif.pkl
    pattern = r'(\w+)_point_\w+_mrc2cif\.pkl'
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        match = re.match(pattern, filename)
        if match:
            pdb_id = match.group(1)
            pdb_ids.add(pdb_id)
    
    # 将PDB ID写入输出文件，每行一个ID
    with open(output_file, 'w') as f:
        for pdb_id in sorted(pdb_ids):
            f.write(f"{pdb_id}\n")
    
    print(f"已成功提取{len(pdb_ids)}个PDB ID，并保存到{output_file}")

# 示例用法
if __name__ == "__main__":
    # 通过命令行参数获取目录路径和输出文件名
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "."  # 如果没有提供参数，则使用当前目录
        
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "pdb_ids.txt"  # 默认输出文件名
    
    print(f"从目录 {directory_path} 提取PDB ID到文件 {output_file}")
    extract_pdb_ids(directory_path, output_file)
