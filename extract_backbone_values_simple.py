import re

# 输入和输出文件路径
input_file = r"C:\Users\Z\Desktop\backbone_contour_levels.txt"
output_file = r"C:\Users\Z\Desktop\simplebackbone_contourlevels.txt"

# 打开输入文件并读取内容
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义提取每个蛋白质信息的模式
pattern = r"PDB代码: ([a-zA-Z0-9]+).*?计算得到的Contour Level: ([0-9.]+)"
matches = re.findall(pattern, content, re.DOTALL)

# 对结果按照PDB代码排序
sorted_matches = sorted(matches, key=lambda x: x[0])

# 准备输出内容，按照PDB代码: 数值的格式
output_content = ""
for pdb_code, contour_level in sorted_matches:
    output_content += f"{pdb_code}: {contour_level}\n"

# 将输出内容写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(output_content)

print(f"提取完成。结果已保存至 {output_file}")
