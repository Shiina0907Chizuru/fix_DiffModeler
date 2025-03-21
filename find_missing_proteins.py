import os

def extract_proteins_from_file1(file_path):
    """
    从第一个文件中提取蛋白质ID列表
    第一个文件格式假设为每行一个蛋白质ID
    """
    proteins = []
    with open(file_path, 'r') as f:
        for line in f:
            # 去除空白字符并添加到列表
            protein_id = line.strip()
            if protein_id:
                proteins.append(protein_id)
    return proteins

def extract_proteins_from_file2(file_path):
    """
    从第二个文件中提取蛋白质ID列表
    第二个文件格式假设为包含蛋白质ID的行
    """
    proteins = []
    with open(file_path, 'r') as f:
        for line in f:
            # 去除空白字符
            line = line.strip()
            # 检查行是否包含蛋白质ID（假设格式为4个字符）
            if line and len(line) >= 4 and line[:4].isalnum():
                # 提取前4个字符作为蛋白质ID
                protein_id = line[:4].lower()
                proteins.append(protein_id)
    return proteins

def main():
    # 文件路径
    file1_path = "C:\\Users\\Z\\Desktop\\20250315.txt"
    file2_path = "C:\\Users\\Z\\Desktop\\20250315contour_level.txt"
    output_path = "C:\\Users\\Z\\Desktop\\20250315missing_proteins.txt"
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"错误：文件 {file1_path} 不存在")
        return
    if not os.path.exists(file2_path):
        print(f"错误：文件 {file2_path} 不存在")
        return
    
    # 读取两个文件中的蛋白质ID
    proteins_file1 = extract_proteins_from_file1(file1_path)
    proteins_file2 = extract_proteins_from_file2(file2_path)
    
    # 转换为集合便于比较
    proteins_set1 = set(proteins_file1)
    proteins_set2 = set(proteins_file2)
    
    # 找出缺失的蛋白质
    missing_proteins = proteins_set1 - proteins_set2
    
    # 输出结果
    print(f"文件1中共有 {len(proteins_set1)} 个蛋白质")
    print(f"文件2中共有 {len(proteins_set2)} 个蛋白质")
    print(f"缺失的蛋白质数量: {len(missing_proteins)}")
    
    # 将缺失的蛋白质写入输出文件
    with open(output_path, 'w') as f:
        for protein in sorted(missing_proteins):
            f.write(f"{protein}\n")
    
    print(f"缺失的蛋白质ID已保存到 {output_path}")

if __name__ == "__main__":
    main()
