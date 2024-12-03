import pandas as pd

# 读取Excel文件
file_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\res_and_cotour.xlsx"  # 使用r前缀避免转义字符问题
df = pd.read_excel(file_path)

# 检查表格中的列名，确保'protein'和'contour'存在
print("表格中的列名:", df.columns)

# 假设表格中有'protein'和'contour'两列
# 提取'protein'和'contour'列
protein_contour_data = df[['protein', 'contour']]

# 打开info.txt文件准备写入
output_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\info.txt"  # 修改为你的输出路径
with open(output_path, 'w') as f:
    for index, row in protein_contour_data.iterrows():
        protein = row['protein']
        contour = row['contour']
        # 写入每一行内容
        f.write(f" {protein}:{contour}\n")

print(f"信息已成功提取并写入 {output_path} 文件.")
