import os

# 指定目录路径
# directory_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\trainpdb_emdb_data"
directory_path = r"/defaultShare/FinialPDB"

# 输出文件路径
output_file = "20250315.txt"

# 获取所有文件夹名称并提取PDB ID
pdb_ids = []
for folder_name in os.listdir(directory_path):
    if folder_name.startswith("PDB-"):
        # 从文件夹名称中提取PDB ID（去掉"PDB-"前缀，并在第一个"-"处停止）
        pdb_id = folder_name.split("-")[1].lower()  # 转换为小写，因为PDB数据库通常使用小写ID
        pdb_ids.append(pdb_id)

# 将PDB ID写入文件
with open(output_file, "w") as f:
    for pdb_id in sorted(pdb_ids):  # 排序以使输出更有组织
        f.write(f"{pdb_id}\n")

print(f"已从{len(pdb_ids)}个文件夹中提取PDB ID并保存到 {output_file}")
