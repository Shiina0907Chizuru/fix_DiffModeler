import os
from data_processing.Unify_Map import Unify_Map
from data_processing.Resize_Map import Resize_Map
from ops.map_utils import increase_map_density
from modeling.map_utils import segment_map

def preprocess_map(input_map_path, save_path, map_name, contour_level=0):
    """
    对输入的 .map 文件进行一系列预处理，包括统一格式、调整大小和密度增加等操作。

    参数:
    - input_map_path (str): 输入 .map 文件的路径。
    - save_path (str): 预处理后文件的保存目录。
    - map_name (str): 文件名的基本名称，用于生成处理后的文件名。
    - contour_level (float): 轮廓阈值，用于调整密度。

    返回:
    - tuple: 预处理后的保存路径和新的 .map 文件路径。
    """
    save_path = os.path.abspath(save_path)

    # 统一 .map 文件格式
    cur_map_path = Unify_Map(input_map_path, os.path.join(save_path, map_name + "_unified.mrc"))

    # 调整 .map 文件大小
    cur_map_path = Resize_Map(cur_map_path, os.path.join(save_path, map_name + ".mrc"))

    # 如果 contour_level 大于 0，则调整密度
    if contour_level > 0:
        cur_map_path = increase_map_density(
            cur_map_path,
            os.path.join(save_path, map_name + "_increase.mrc"),
            contour_level
        )
        contour_level = 0  # 将 contour_level 设为 0，用于下一步的处理

    # 生成分割后的 .map 文件
    new_map_path = os.path.join(save_path, map_name + "_segment.mrc")
    segment_map(cur_map_path, new_map_path, contour=contour_level)

    return save_path, new_map_path




# 设置蛋白质名称
protein_name = "8h3r"  # 只需修改此处的蛋白质名称即可，例如 "3j22" 或 "8h3r"

# 根据蛋白质名称自动设置路径和轮廓阈值
input_map_path = rf"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\{protein_name}\emd.map"
save_path = rf"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\{protein_name}\processed"
map_name = protein_name

# 根据蛋白质名称自动设置 contour_level
contour_levels = {
    "3j9p": 8.0,
    "3j22": 1.0,
    "8h3r": 0.001
}
contour_level = contour_levels.get(protein_name, 1.0)  # 默认值为 1.0，如果未找到匹配的名称

# 调用预处理函数
processed_save_path, processed_map_path = preprocess_map(input_map_path, save_path, map_name, contour_level)

print(f"Processed files are saved at: {processed_save_path}")
print(f"New processed map path: {processed_map_path}")
