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
from ops.os_operation import mkdir, clean_directory

def generate_dataset(input_map_path, label_map_path, save_dir, contour, box_size=64, stride=32):
    """
    根据输入的 .map 文件和标签 .mrc 文件生成数据集。
    """
    # 创建保存路径（如果不存在）
    mkdir(save_dir)
    
    # 清空保存目录中的所有文件
    clean_directory(save_dir)
    print(f"已清空目录: {save_dir}")

    # 读取并处理输入 .map 文件
    params = {"model": {"diffusion": {"box_size": box_size, "stride": stride}}}
    # 直接使用save_dir作为输入和输出的保存路径
    Coord_Voxel, map_data, adjusted_contour = generate_infer_data(input_map_path, save_dir, contour, params)
    
    # 分割标签 .mrc 文件并保存到同一目录
    with mrcfile.open(label_map_path, permissive=True) as label_mrc:
        label_data = np.array(label_mrc.data)
        
        # 输出归一化前的统计信息
        min_value = np.min(label_data)
        max_value = np.max(label_data)
        print(f"标签数据归一化前：最小值={min_value:.6f}, 最大值={max_value:.6f}")
        print(f"标签数据负值比例：{np.sum(label_data < 0) / label_data.size * 100:.2f}%")
        
        # 执行最大最小值归一化
        if max_value != min_value:  # 避免除零错误
            label_data = (label_data - min_value) / (max_value - min_value)
        else:
            label_data = np.zeros_like(label_data)
            
        # 输出归一化后的统计信息
        print(f"标签数据归一化后：最小值={np.min(label_data):.6f}, 最大值={np.max(label_data):.6f}")
        
        for i, (x, y, z) in enumerate(Coord_Voxel):
            x_end = min(x + box_size, label_data.shape[0])
            y_end = min(y + box_size, label_data.shape[1])
            z_end = min(z + box_size, label_data.shape[2])
            segment_label = np.zeros((box_size, box_size, box_size))
            segment_label[:x_end - x, :y_end - y, :z_end - z] = label_data[x:x_end, y:y_end, z:z_end]
            output_path = os.path.join(save_dir, f"output_{i}.npy")
            np.save(output_path, segment_label)
            
        print(f"已保存 {len(Coord_Voxel)} 个输入切片和 {len(Coord_Voxel)} 个输出切片")
    print("Dataset generated successfully.")

# 原始版本的函数（已注释）
"""
def process_from_file(input_file):
    '''
    从文本文件中读取蛋白质信息并生成数据集。
    '''
    base_path = "/share/home/xiaogenz/users/jiangzhaox/DiffModeler_data/43_proteindataset"

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or ':' not in line:
                continue

            try:
                protein_name, contour_level = line.split(':')
                contour_level = float(contour_level)

                # 设置路径
                input_map_path = os.path.join(base_path, protein_name, f"{protein_name}_segment.mrc")
                label_map_path = os.path.join(base_path, protein_name, f"{protein_name}_label.mrc")
                save_dir = os.path.join(base_path, protein_name, "Dataset")

                # 调用数据集生成函数
                generate_dataset(input_map_path, label_map_path, save_dir, contour_level)
            except ValueError:
                print(f"Invalid line format: {line}")
                continue
"""

# 新版本的函数
def process_from_file(input_file, skip_existing=True):
    """
    从文本文件中读取蛋白质信息并生成数据集。
    适配新的目录结构版本。
    
    Args:
        input_file (str): 包含蛋白质名称和等值面水平的文本文件路径
        skip_existing (bool): 是否跳过已有backbone_Dataset文件夹的蛋白质，默认为True
    """
    # base_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\newdateset\trainpdb_emdb_data"
    base_path = r"/zhaoxuanj/FinialPDB"

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or ':' not in line:
                continue

            try:
                protein_name, contour_level = line.split(':')
                contour_level = float(contour_level)
                protein_folder = f"PDB-{protein_name.lower()}-EMD-*"  # 使用通配符匹配EMD编号
                
                # 使用glob找到匹配的文件夹
                import glob
                matching_folders = glob.glob(os.path.join(base_path, protein_folder))
                if not matching_folders:
                    print(f"No matching folder found for protein: {protein_name}")
                    continue
                    
                protein_dir = matching_folders[0]  # 使用第一个匹配的文件夹
                
                # 设置输入和输出路径
                input_map_path = os.path.join(protein_dir, "processed", f"{protein_name}_segment.mrc")
                label_map_path = os.path.join(protein_dir, "processed", f"{protein_name}_backbone.mrc")
                save_dir = os.path.join(protein_dir, "backbone_Dataset")
                
                # 检查是否已有backbone_Dataset文件夹，如果有且skip_existing为True，就跳过该蛋白质
                if skip_existing and os.path.exists(save_dir):
                    print(f"检测到已有数据集文件夹，跳过处理: {save_dir}")
                    continue

                # 确保输出目录存在
                mkdir(save_dir)

                print(f"Processing {protein_name}...")
                print(f"Input map: {input_map_path}")
                print(f"Label map: {label_map_path}")
                print(f"Save directory: {save_dir}")

                # 调用数据集生成函数
                generate_dataset(input_map_path, label_map_path, save_dir, contour_level)
                print(f"Finished processing {protein_name}")
                
            except Exception as e:
                print(f"Error processing {protein_name}: {str(e)}")
                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate protein backbone datasets from map files.')
    parser.add_argument("--info_txt", type=str, required=True, help="Path to the input text file containing protein names and contour levels.")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="跳过已有backbone_Dataset文件夹的蛋白质（默认行为）")
    parser.add_argument("--no_skip", dest="skip_existing", action="store_false", help="不跳过已有backbone_Dataset文件夹的蛋白质，强制重新生成")
    args = parser.parse_args()
    process_from_file(args.info_txt, args.skip_existing)
