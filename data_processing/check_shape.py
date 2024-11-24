import os
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 文件夹路径（请根据实际情况修改路径）
folder_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\mydataset\3j22"

# 遍历文件夹，检测文件维度
def check_npy_files_with_progress(folder_path):
    # 获取所有 .npy 文件
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} .npy files.")

    # 遍历文件并检查维度，使用 tqdm 显示进度条
    for idx, file_name in enumerate(tqdm(npy_files, desc="Processing files")):
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.load(file_path)  # 加载 .npy 文件
            if len(data.shape) != 3:  # 检查维度是否为 3
                print(f"File with incorrect dimensions: {file_name}, shape: {data.shape}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# 运行检测
check_npy_files_with_progress(folder_path)
