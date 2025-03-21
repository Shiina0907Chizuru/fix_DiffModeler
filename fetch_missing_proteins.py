import os
import sys
import subprocess
import time
import shutil

def extract_proteins_from_file1(file_path):
    """从第一个文件中提取蛋白质ID列表"""
    proteins = []
    with open(file_path, 'r') as f:
        for line in f:
            protein_id = line.strip()
            if protein_id:
                proteins.append(protein_id.lower())
    return proteins

def extract_proteins_from_file2(file_path):
    """从第二个文件中提取已处理的蛋白质ID列表"""
    proteins = []
    if not os.path.exists(file_path):
        return proteins
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                # 假设格式为 "XXXX: ..."，提取冒号前的ID
                protein_id = line.split(":")[0].strip().lower()
                if protein_id and len(protein_id) >= 4:
                    proteins.append(protein_id)
    return proteins

def create_temp_pdb_list(proteins, output_path):
    """创建临时PDB列表文件，包含缺失的蛋白质"""
    with open(output_path, 'w') as f:
        for protein in proteins:
            f.write(f"{protein}\n")
    return len(proteins)

def create_modified_script(original_script, temp_script, temp_pdb_list_path):
    """创建临时脚本副本并修改PDB列表路径"""
    with open(original_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换PDB列表文件路径，使用正斜杠避免转义问题
    normalized_path = temp_pdb_list_path.replace('\\', '/')
    
    modified_content = content.replace(
        'pdb_list_file = "c:/Users/Z/Desktop/20250315.txt"',
        f'pdb_list_file = "{normalized_path}"'
    )
    
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(modified_content)

def main():
    # 文件路径，使用正斜杠或者原始字符串表示
    original_pdb_list = "C:/Users/Z/Desktop/20250315.txt"
    temp_output_file = "C:/Users/Z/Desktop/20250315contour_level&resolution_temp.txt"
    temp_pdb_list = "C:/Users/Z/Desktop/20250315_missing.txt"
    original_script = "e:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler/get_contour_levels_and_resolution.py"
    temp_script = "e:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler/get_contour_levels_temp.py"
    
    max_attempts = 5  # 最大尝试次数
    
    attempt = 1
    while attempt <= max_attempts:
        print(f"\n===== 尝试 #{attempt} =====")
        
        # 读取两个文件中的蛋白质ID
        proteins_file1 = extract_proteins_from_file1(original_pdb_list)
        proteins_file2 = extract_proteins_from_file2(temp_output_file)
        
        # 转换为集合便于比较
        proteins_set1 = set(proteins_file1)
        proteins_set2 = set(proteins_file2)
        
        # 找出缺失的蛋白质
        missing_proteins = proteins_set1 - proteins_set2
        missing_list = sorted(list(missing_proteins))
        
        # 输出结果
        print(f"文件1中共有 {len(proteins_set1)} 个蛋白质")
        print(f"文件2中共有 {len(proteins_set2)} 个蛋白质")
        print(f"缺失的蛋白质数量: {len(missing_proteins)}")
        
        # 如果没有缺失的蛋白质，则完成
        if not missing_proteins:
            print("所有蛋白质信息已获取完成！")
            break
        
        # 创建临时PDB列表
        batch_size = min(50, len(missing_list))  # 每次处理最多50个蛋白质
        current_batch = missing_list[:batch_size]
        
        print(f"正在准备获取 {batch_size} 个蛋白质信息...")
        count = create_temp_pdb_list(current_batch, temp_pdb_list)
        
        # 创建修改过的临时脚本
        create_modified_script(original_script, temp_script, temp_pdb_list)
        
        # 运行修改后的爬虫脚本获取信息
        print(f"启动爬虫脚本处理 {count} 个蛋白质...")
        try:
            # 使用临时脚本
            command = [
                sys.executable,  # 当前Python解释器
                temp_script
            ]
            
            # 执行爬虫命令
            process = subprocess.Popen(command)
            process.wait()
            
            print("爬虫脚本执行完成，等待5秒后检查结果...")
            time.sleep(5)  # 等待文件写入完成
            
        except Exception as e:
            print(f"运行爬虫脚本时出错: {e}")
        
        attempt += 1
    
    # 清理临时文件
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    if attempt > max_attempts and missing_proteins:
        print(f"达到最大尝试次数 ({max_attempts})，仍有 {len(missing_proteins)} 个蛋白质信息未获取。")
        # 将剩余缺失的蛋白质保存到文件
        with open("C:/Users/Z/Desktop/20250315_remaining_missing.txt", 'w') as f:
            for protein in sorted(missing_proteins):
                f.write(f"{protein}\n")
        print("剩余缺失的蛋白质ID已保存到 C:/Users/Z/Desktop/20250315_remaining_missing.txt")
    
if __name__ == "__main__":
    main()
