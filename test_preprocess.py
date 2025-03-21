import os
import mrcfile
import numpy as np
from processed_map import preprocess_map

# 测试参数
input_map_path = r"C:\Users\Z\Desktop\EMD-34655.map"
save_dir = r"C:\Users\Z\Desktop\test_result"
map_name = "EMD-34655"
contour_level = 0.4

# 确保输出目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 记录原始文件信息
print(f"处理文件: {input_map_path}")
print(f"设置的Contour值: {contour_level}")

with mrcfile.open(input_map_path, permissive=True) as mrc:
    orig_data = mrc.data.copy()
    orig_mean = np.mean(orig_data)
    orig_std = np.std(orig_data)
    orig_min = np.min(orig_data)
    orig_max = np.max(orig_data)
    print(f"\n原始数据统计:")
    print(f"  均值: {orig_mean:.6f}")
    print(f"  标准差: {orig_std:.6f}")
    print(f"  最小值: {orig_min:.6f}")
    print(f"  最大值: {orig_max:.6f}")

# 运行预处理
print("\n开始处理...")
save_path, new_map_path = preprocess_map(input_map_path, save_dir, map_name, contour_level)

# 分析处理后的文件
print(f"\n生成的segment文件: {new_map_path}")
with mrcfile.open(new_map_path, permissive=True) as mrc:
    seg_data = mrc.data
    seg_mean = np.mean(seg_data)
    seg_std = np.std(seg_data)
    seg_min = np.min(seg_data)
    seg_max = np.max(seg_data)
    print(f"\nSegment数据统计:")
    print(f"  均值: {seg_mean:.6f}")
    print(f"  标准差: {seg_std:.6f}")
    print(f"  最小值: {seg_min:.6f}")
    print(f"  最大值: {seg_max:.6f}")
    print(f"  数据形状: {seg_data.shape}")

# 输出其他生成的中间文件
print("\n生成的所有文件:")
result_files = os.listdir(save_dir)
for file in result_files:
    print(f"  {file}")

print(f"\n处理完成，结果保存在: {save_dir}")
