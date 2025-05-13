import os
import sys
import re

def extract_contour_values(pdb_ids_file, contour_info_file, output_file):
    """
    从contour信息文件中提取指定蛋白质ID的contour阈值
    
    参数:
        pdb_ids_file: 包含蛋白质ID列表的文件路径
        contour_info_file: 包含contour信息的文件路径
        output_file: 输出文件路径
    """
    # 读取蛋白质ID列表
    if not os.path.exists(pdb_ids_file):
        print(f"错误: 蛋白质ID文件 {pdb_ids_file} 不存在")
        return False
    
    with open(pdb_ids_file, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    if not pdb_ids:
        print(f"错误: 蛋白质ID文件 {pdb_ids_file} 为空")
        return False
    
    print(f"已读取 {len(pdb_ids)} 个蛋白质ID")
    
    # 读取contour信息文件
    if not os.path.exists(contour_info_file):
        print(f"错误: contour信息文件 {contour_info_file} 不存在")
        return False
    
    # 存储提取的contour值
    contour_values = {}
    
    with open(contour_info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 预期格式: "蛋白质ID: 阈值" (如 "6c24: 0.0383")
            match = re.match(r'(\w+):\s*([0-9.]+)', line)
            if match:
                pdb_id = match.group(1)
                contour = match.group(2)
                contour_values[pdb_id] = contour
    
    print(f"已从contour文件中提取 {len(contour_values)} 个条目")
    
    # 提取目标蛋白质的contour值
    found_count = 0
    results = []
    
    for pdb_id in pdb_ids:
        if pdb_id in contour_values:
            results.append(f"{pdb_id}: {contour_values[pdb_id]}")
            found_count += 1
    
    # 将结果写入输出文件
    with open(output_file, 'w') as f:
        f.write("# 蛋白质ID: 阈值\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"已成功提取 {found_count}/{len(pdb_ids)} 个蛋白质的contour值，并保存到 {output_file}")
    
    # 如果有未找到contour值的蛋白质，输出它们的ID
    if found_count < len(pdb_ids):
        not_found = [pdb_id for pdb_id in pdb_ids if pdb_id not in contour_values]
        print(f"以下 {len(not_found)} 个蛋白质未找到contour值:")
        for pdb_id in not_found[:10]:  # 只打印前10个，避免输出过多
            print(f"  {pdb_id}")
        if len(not_found) > 10:
            print(f"  ...以及其他 {len(not_found)-10} 个")
    
    return True

if __name__ == "__main__":
    # 从命令行参数获取文件路径
    if len(sys.argv) < 3:
        print("用法: python extract_contour_values.py 蛋白质ID文件 contour信息文件 [输出文件]")
        print("例如: python extract_contour_values.py pdb_ids.txt contour_info.txt output.txt")
        sys.exit(1)
    
    pdb_ids_file = sys.argv[1]
    contour_info_file = sys.argv[2]
    
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        output_file = "extracted_contour_values.txt"
    
    extract_contour_values(pdb_ids_file, contour_info_file, output_file)
