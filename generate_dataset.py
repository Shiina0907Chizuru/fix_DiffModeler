# import mrcfile
# import numpy as np
# import os
# from data_processing.generate_input_data import gen_input_box, generate_infer_data,gen_output_box
# from ops.os_operation import mkdir
#
#
# # def generate_dataset(input_map_path, label_map_path, save_dir, contour, box_size=32, stride=16):
# #     # 创建保存路径
# #     mkdir(save_dir)
# #
# #     # 读取并处理输入 .map 文件
# #     params = {"model": {"diffusion": {"box_size": box_size, "stride": stride}}}
# #     Coord_Voxel, map_data, adjusted_contour = generate_infer_data(input_map_path, save_dir, contour, params)
# #
# #     # 保存输入数据块
# #     input_save_dir = os.path.join(save_dir, "inputs")
# #     mkdir(input_save_dir)
# #     gen_input_box(map_data, box_size, stride, adjusted_contour, input_save_dir)
# #
# #     # 读取并处理标签 .mrc 文件
# #     with mrcfile.open(label_map_path, permissive=True) as label_mrc:
# #         label_data = np.array(label_mrc.data)
# #
# #     # 保存标签数据块
# #     output_save_dir = os.path.join(save_dir, "outputs")
# #     mkdir(output_save_dir)
# #     gen_output_box(label_data, box_size, stride, adjusted_contour, output_save_dir)
# #
# #     print("Dataset generated successfully.")
# def generate_dataset(input_map_path, label_map_path, save_dir, contour, box_size=64, stride=32):
#     # 创建保存路径
#     mkdir(save_dir)
#
#     # 读取并处理输入 .map 文件
#     params = {"model": {"diffusion": {"box_size": box_size, "stride": stride}}}
#     Coord_Voxel, map_data, adjusted_contour = generate_infer_data(input_map_path, save_dir, contour, params)
#
#     # #保存输入数据块
#     # input_save_dir = os.path.join(save_dir, "inputs")
#     # mkdir(input_save_dir)
#     # gen_input_box(map_data, box_size, stride, adjusted_contour, input_save_dir)
#
#     # 分割标签 .mrc 文件
#     with mrcfile.open(label_map_path, permissive=True) as label_mrc:
#         label_data = np.array(label_mrc.data)
#         # label_save_dir = os.path.join(save_dir, "outputs")
#         label_save_dir = os.path.join(save_dir)
#         mkdir(label_save_dir)
#
#         for i, (x, y, z) in enumerate(Coord_Voxel):
#             x_end = min(x + box_size, label_data.shape[0])
#             y_end = min(y + box_size, label_data.shape[1])
#             z_end = min(z + box_size, label_data.shape[2])
#             segment_label = np.zeros((box_size, box_size, box_size))
#             segment_label[:x_end - x, :y_end - y, :z_end - z] = label_data[x:x_end, y:y_end, z:z_end]
#             # if segment_label.ndim != 3:
#             #     print(f"Warning: segment_label at index {i} has dimension {segment_label.shape}")
#             output_path = os.path.join(label_save_dir, f"output_{i}.npy")
#             np.save(output_path, segment_label)
#         print(f"In total we prepared {len(Coord_Voxel)} boxes as output")
#     print("Dataset generated successfully.")
#
# # 设置蛋白质名称
# protein_name = "3j9p"  # 只需更改此处的蛋白质名称即可，例如 "3j9p" 或 "3j22"
#
# # 根据蛋白质名称自动设置路径和轮廓阈值
# base_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\DATA"
# input_map_path = os.path.join(base_path, protein_name, "processed", f"{protein_name}_segment.mrc")
# label_map_path = os.path.join(base_path, protein_name, "label_closest_voxel.mrc")
# save_dir = os.path.join(base_path, protein_name, "Dataset")
#
# # 根据蛋白质名称自动设置 contour_level
# contour_levels = {
#     "3j9p": 8.0,
#     "3j22": 1.0,
#     "8h3r": 0.001
# }
# contour = contour_levels.get(protein_name, 1.0)  # 默认值为 1.0，如果未找到匹配的名称
#
# generate_dataset(input_map_path, label_map_path, save_dir, contour)

# 集群版本
import os
import mrcfile
import numpy as np
from data_processing.generate_input_data import generate_infer_data
from ops.os_operation import mkdir

def generate_dataset(input_map_path, label_map_path, save_dir, contour, box_size=64, stride=32):
    """
    根据输入的 .map 文件和标签 .mrc 文件生成数据集。
    """
    # 创建保存路径
    mkdir(save_dir)

    # 读取并处理输入 .map 文件
    params = {"model": {"diffusion": {"box_size": box_size, "stride": stride}}}
    Coord_Voxel, map_data, adjusted_contour = generate_infer_data(input_map_path, save_dir, contour, params)

    # 分割标签 .mrc 文件
    with mrcfile.open(label_map_path, permissive=True) as label_mrc:
        label_data = np.array(label_mrc.data)
        label_save_dir = os.path.join(save_dir)
        mkdir(label_save_dir)

        for i, (x, y, z) in enumerate(Coord_Voxel):
            x_end = min(x + box_size, label_data.shape[0])
            y_end = min(y + box_size, label_data.shape[1])
            z_end = min(z + box_size, label_data.shape[2])
            segment_label = np.zeros((box_size, box_size, box_size))
            segment_label[:x_end - x, :y_end - y, :z_end - z] = label_data[x:x_end, y:y_end, z:z_end]
            output_path = os.path.join(label_save_dir, f"output_{i}.npy")
            np.save(output_path, segment_label)

        print(f"In total we prepared {len(Coord_Voxel)} boxes as output")
    print("Dataset generated successfully.")

def process_from_file(input_file):
    """
    从文本文件中读取蛋白质信息并生成数据集。
    """
    base_path = "/share/home/xiaogenz/users/jiangzhaox/DiffModeler_data/mydataset"

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or ':' not in line:
                continue

            try:
                protein_name, contour_level = line.split(':')
                contour_level = float(contour_level)

                # 设置路径
                input_map_path = os.path.join(base_path, protein_name, "processed", f"{protein_name}_segment.mrc")
                label_map_path = os.path.join(base_path, protein_name, "label_closest_voxel.mrc")
                save_dir = os.path.join(base_path, protein_name, "Dataset")

                # 调用数据集生成函数
                generate_dataset(input_map_path, label_map_path, save_dir, contour_level)
            except ValueError:
                print(f"Invalid line format: {line}")
                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate datasets for protein structures.")
    parser.add_argument("--info_txt", type=str, required=True, help="Path to the input text file containing protein names and contour levels.")
    args = parser.parse_args()
    process_from_file(args.info_txt)
