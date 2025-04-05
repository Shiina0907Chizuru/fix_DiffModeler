#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用DiffModeler的preprocess_map函数处理EMD-6824蛋白质密度图
"""

import os
import sys
from processed_map import preprocess_map

# 输入和输出路径设置
input_map_path = r"C:\Users\Z\Desktop\emd_6824.map"
protein_code = "6824"  # 从输入文件名中提取的蛋白质代码
output_dir = r"PDB-{}-EMD-{}".format(protein_code, protein_code)  # 按照DiffModeler约定命名
processed_dir = os.path.join(output_dir, "processed")  # 处理后的文件放在processed子目录

# 创建输出目录
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# 设置轮廓阈值
contour_level = 3.0

# 调用preprocess_map函数
print("开始预处理密度图: {}".format(input_map_path))
print("轮廓阈值(contour level): {}".format(contour_level))
print("输出目录: {}".format(processed_dir))

try:
    save_path, output_map_path = preprocess_map(
        input_map_path, 
        processed_dir, 
        protein_code, 
        contour_level=contour_level
    )
    
    print("\n预处理完成!")
    print("输出路径: {}".format(save_path))
    print("处理后的密度图: {}".format(output_map_path))
    
except Exception as e:
    print("\n处理过程中出错:")
    print(str(e))
    
print("\n提示: 如需使用DiffModeler分析此蛋白质，请确保目录结构符合项目约定。")
