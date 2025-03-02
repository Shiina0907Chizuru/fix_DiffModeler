import os
import numpy as np
import sys

def check_empty_files(folder_path):
    """检查文件夹中的input*.npy和output*.npy文件是否存在全空样本"""
    print(f"\n检查文件夹: {folder_path}")
    
    # 获取所有npy文件并分类
    output_files = [f for f in os.listdir(folder_path) if f.startswith('output') and f.endswith('.npy')]
    input_files = [f for f in os.listdir(folder_path) if f.startswith('input') and f.endswith('.npy')]
    
    # 检查output文件
    empty_outputs = []
    empty_output_numbers = []
    for file in output_files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        if np.all(data == 0):  # 检查是否全为0
            empty_outputs.append(file)
            num = file.replace('output_', '').replace('.npy', '')
            empty_output_numbers.append(num)
            print(f"发现全空output文件: {file}")
    
    # 检查input文件
    empty_inputs = []
    empty_input_numbers = []
    for file in input_files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        if np.all(data == 0):  # 检查是否全为0
            empty_inputs.append(file)
            num = file.replace('input_', '').replace('.npy', '')
            empty_input_numbers.append(num)
            print(f"发现全空input文件: {file}")
    
    # 输出统计信息
    print(f"\n共发现 {len(empty_outputs)} 个全空output文件")
    if empty_outputs:
        print("\n全空output文件编号列表:")
        print(','.join(sorted(empty_output_numbers, key=lambda x: int(x))))
    
    print(f"\n共发现 {len(empty_inputs)} 个全空input文件")
    if empty_inputs:
        print("\n全空input文件编号列表:")
        print(','.join(sorted(empty_input_numbers, key=lambda x: int(x))))
        
    # 检查是否有同时为空的文件
    common_numbers = set(empty_input_numbers) & set(empty_output_numbers)
    if common_numbers:
        print(f"\n发现 {len(common_numbers)} 对input和output同时为空的文件，编号:")
        print(','.join(sorted(list(common_numbers), key=lambda x: int(x))))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python check_empty_output.py <文件夹路径>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        print(f"错误: 路径 {folder_path} 不存在!")
        sys.exit(1)
        
    check_empty_files(folder_path)
