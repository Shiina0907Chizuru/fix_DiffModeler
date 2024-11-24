import os
import sys
from ops.pdb_utils import read_proteinpdb_data
from ops.io_utils import write_pickle
from ops.map_utils import process_map_data,save_label_map
import numpy as np
from ops.map_coord_utils import permute_ns_coord_to_pdb,permute_pdb_coord_to_map
import mrcfile
from modeling.pdb_utils import remove_hetatm_lines
sys.path.append('/ziyingz/Programs/DiffModeler')


def assign_label_special(map_data, mapc, mapr, maps, origin ,nxstart,nystart,nzstart,cif_info_dict):
    #here nstart is useless, must removed because pdb2vol's generated map
    """
    :param map_data: map array
    :param mapc: map column indicator
    :param mapr: map row indicator
    :param maps: map section indicator
    :param origin: starting point of map, x, y,z
    :param cif_info_dict:
    :return:
    """
    output_atom_label = np.zeros(map_data.shape)#-1 denotes background
    distance_array = np.ones(map_data.shape) * 100 #record the closest distance
    count_check = 0
    for tmp_id in cif_info_dict:
        tmp_dict = cif_info_dict[tmp_id]
        current_coord = tmp_dict['coord']
        current_atom = tmp_dict['atom']
        current_radius =tmp_dict['radius']
        print(current_radius)
        current_nuc = tmp_dict['nuc']
        # nstart = [nxstart,nystart,nzstart]
        # nstart = permute_ns_coord_to_pdb(nstart,mapc,mapr,maps)
        output_coord = []
        for k in range(3):
            output_coord.append(current_coord[k]-origin[k])
        new_x, new_y, new_z = permute_pdb_coord_to_map(output_coord,mapc,mapr,maps)
        print("input x %.4f, y %.4f, z %.4f"%(output_coord[0],output_coord[1],output_coord[2]))
        print("output x %.4f, y %.4f, z %.4f"%(new_x, new_y, new_z))
        check_radius = int(np.ceil(current_radius))
        int_x, int_y,int_z = int(new_x),int(new_y),int(new_z)
        square_limit = current_radius ** 2
        for ii in range(int_x-check_radius,int_x+check_radius+1):
            if ii >= distance_array.shape[0]:
                continue
            for jj in range(int_y-check_radius,int_y+check_radius+1):
                if jj >=distance_array.shape[1]:
                    continue
                for kk in range(int_z-check_radius,int_z+check_radius+1):
                    if kk>=distance_array.shape[2]:
                        continue
                    cur_distance = (ii-new_x)**2+(jj-new_y)**2+(kk-new_z)**2
                    if cur_distance<= square_limit:
                        record_distance = distance_array[ii,jj,kk]
                        if record_distance>=cur_distance:
                            #update record
                            distance_array[ii,jj,kk] = cur_distance
                            output_atom_label[ii,jj,kk] = current_atom
        count_check+=1
        if count_check%100==0:
            print("Calcu finished %d/%d"%(count_check,len(cif_info_dict)))
    return distance_array,output_atom_label
    


def assign_label_closest_voxel(map_data, mapc, mapr, maps, origin, cif_info_dict):
    """
    仅对每个原子标记一个最近的体素
    :param map_data: map array
    :param mapc, mapr, maps: map column, row, and section indicators
    :param origin: 起点坐标 (x, y, z)
    :param cif_info_dict: 原子信息字典
    :return: 输出标签数组
    """
    output_atom_label = np.zeros(map_data.shape)  # 初始标签数组，0表示背景
    distance_array = np.ones(map_data.shape) * 100  # 初始距离数组，用于记录最小距离

    for tmp_id, tmp_dict in cif_info_dict.items():
        # 获取当前原子的信息
        current_coord = tmp_dict['coord']
        current_atom = tmp_dict['atom']
        output_coord = [current_coord[k] - origin[k] for k in range(3)]  # 计算网格坐标

        # 将坐标转换到 map_data 的网格系统中
        new_x, new_y, new_z = permute_pdb_coord_to_map(output_coord, mapc, mapr, maps)

        # 找到最接近的整数体素坐标
        int_x, int_y, int_z = int(round(new_x)), int(round(new_y)), int(round(new_z))

        # 检查边界
        if 0 <= int_x < map_data.shape[0] and 0 <= int_y < map_data.shape[1] and 0 <= int_z < map_data.shape[2]:
            # 计算到当前体素点的距离
            cur_distance = (new_x - int_x) ** 2 + (new_y - int_y) ** 2 + (new_z - int_z) ** 2

            # 更新最近体素点的标签和距离
            if cur_distance < distance_array[int_x, int_y, int_z]:
                distance_array[int_x, int_y, int_z] = cur_distance
                output_atom_label[int_x, int_y, int_z] = current_atom

    return distance_array, output_atom_label

############################################################将密度图中对应PDB区域地方保留，其余都去除##########################################################################
def mask_map_by_pdb_slow(input_map_path,output_map_path,final_pdb_output,cutoff=2,keep_label=True):
    """

    :param input_map_path:
    :param output_map_path:
    :param final_pdb_output:
    :param cutoff:
    :param keep_label: keep labeled region, or keep unlabeled region.
    :return:
    """
    output_dir = os.path.split(output_map_path)[0]
    pdb_info_dict = read_proteinpdb_data(final_pdb_output,run_type=2,atom_cutoff=cutoff)
    pdb_info_path = os.path.join(output_dir,"maskprotein_info_%d.pkl"%cutoff)
    write_pickle(pdb_info_dict,pdb_info_path)
    #pickle.dump(pdb_info_dict, open(pdb_info_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    map_data, mapc, mapr, maps, origin,nxstart,nystart,nzstart = process_map_data(input_map_path)
    # distance_array, output_atom_label = \
                # assign_label_special(map_data, mapc, mapr, maps, origin,nxstart,nystart,nzstart,  pdb_info_dict)
    distance_array, output_atom_label = \
                assign_label_closest_voxel(map_data, mapc, mapr, maps, origin,  pdb_info_dict)            
               
                
    map_data = np.array(map_data)
    # count_meaningful = len(map_data[map_data!=0])
    # count_masked = len(output_atom_label[output_atom_label!=0])
    # print("mask %d grid points out of %d meaningful grid points. "
    #       "Ratio:%.2f"%(count_masked,count_meaningful,count_masked/count_meaningful))
    if keep_label:
        #unassigned region all set to 0, only focused on the labeled region
        map_data[output_atom_label==0]=0
    else:
        map_data[output_atom_label!=0]=0
    save_label_map(input_map_path,output_map_path,output_atom_label)
from TEMPy import *
from TEMPy.protein.structure_parser import *
from TEMPy.protein.structure_blurrer import *
from TEMPy.protein.scoring_functions import *
from TEMPy.maps.map_parser import *

def mask_map_by_pdb(input_map_path,output_map_path,final_pdb_output,keep_label=False):
    """

    :param input_map_path:
    :param output_map_path:
    :param final_pdb_output:
    :param cutoff:
    :param keep_label: keep labeled region, or keep unlabeled region.
    :return:
    """
    #first use pdb to blue simulated map
    if keep_label:
        resolution =10#includes bigger map as possible
    else:
        resolution = 2
    sb = StructureBlurrer()
    remove_hetatm_lines(final_pdb_output, final_pdb_output)  # avoid issue by tempy
    pdb1 = PDBParser.read_PDB_file('PDB1',final_pdb_output,hetatm=True,water=True)
    map1 = MapParser.readMRC(input_map_path)
    sim_map = sb.gaussian_blur_real_space(prot=pdb1,densMap=map1,resolution=resolution)
    bin_map1 = map1.fullMap
    bin_map2 = sim_map.fullMap
    if keep_label:
        mask = bin_map2>=0.01
    else:
        mask = bin_map2<0.001
    bin_map1 = bin_map1*mask
    map1.fullMap = bin_map1
    map1.write_to_MRC_file(output_map_path)



def format_map(input_map,output_map):
    """

    :param input_map:
    :param output_map:
    :return:
    generate a map with correct header, to fix the machine stamp error
    """
    with mrcfile.open(input_map, permissive=True) as mrc:
        prev_voxel_size = mrc.voxel_size
        prev_voxel_size_x = float(prev_voxel_size['x'])
        prev_voxel_size_y = float(prev_voxel_size['y'])
        prev_voxel_size_z = float(prev_voxel_size['z'])
        nx, ny, nz, nxs, nys, nzs, mx, my, mz = \
            mrc.header.nx, mrc.header.ny, mrc.header.nz, \
            mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart, \
            mrc.header.mx, mrc.header.my, mrc.header.mz
        orig = mrc.header.origin
        #check the useful density in the input

        new_data = np.array(mrc.data)

        mrc_new = mrcfile.new(output_map, data=new_data, overwrite=True)
        vsize = mrc_new.voxel_size
        vsize.flags.writeable = True
        vsize.x = prev_voxel_size_x
        vsize.y = prev_voxel_size_y
        vsize.z = prev_voxel_size_z
        mrc_new.voxel_size = vsize
        mrc_new.update_header_from_data()
        mrc_new.header.nx = nx
        mrc_new.header.ny = ny
        mrc_new.header.nz = nz
        mrc_new.header.nxstart = 0#nxs
        mrc_new.header.nystart = 0#nys
        mrc_new.header.nzstart = 0#nzs
        mrc_new.header.mx = mx
        mrc_new.header.my = my
        mrc_new.header.mz = mz
        mrc_new.header.mapc = mrc.header.mapc
        mrc_new.header.mapr = mrc.header.mapr
        mrc_new.header.maps = mrc.header.maps

        mrc_new.header.origin = orig
        mrc_new.update_header_stats()
        mrc.print_header()
        mrc_new.print_header()
        mrc_new.close()
    return output_map

# from data_processing.Unify_Map import Unify_Map
# from data_processing.Resize_Map import Resize_Map
# from ops.map_utils import increase_map_density
# from modeling.map_utils import segment_map
#
# def preprocess_map(input_map_path, save_path, map_name, contour_level=0):
#     """
#     对输入的 .map 文件进行一系列预处理，包括统一格式、调整大小和密度增加等操作。
#
#     参数:
#     - input_map_path (str): 输入 .map 文件的路径。
#     - save_path (str): 预处理后文件的保存目录。
#     - map_name (str): 文件名的基本名称，用于生成处理后的文件名。
#     - contour_level (float): 轮廓阈值，用于调整密度。
#
#     返回:
#     - tuple: 预处理后的保存路径和新的 .map 文件路径。
#     """
#     save_path = os.path.abspath(save_path)
#
#     # 统一 .map 文件格式
#     cur_map_path = Unify_Map(input_map_path, os.path.join(save_path, map_name + "_unified.mrc"))
#
#     # 调整 .map 文件大小
#     cur_map_path = Resize_Map(cur_map_path, os.path.join(save_path, map_name + ".mrc"))
#
#     # 如果 contour_level 大于 0，则调整密度
#     if contour_level < 0:
#         cur_map_path = increase_map_density(
#             cur_map_path,
#             os.path.join(save_path, map_name + "_increase.mrc"),
#             contour_level
#         )
#         contour_level = 0  # 将 contour_level 设为 0，用于下一步的处理
#
#     # 生成分割后的 .map 文件
#     new_map_path = os.path.join(save_path, map_name + "_segment.mrc")
#     segment_map(cur_map_path, new_map_path, contour=contour_level)
#
#     return save_path, new_map_path




#mask_map_by_pdb("/ziyingz/library/43cryo_Benchmark/43map/6v9i/segemdcoutour0.5voxelsize1.map","/ziyingz/library/43cryo_Benchmark/43map/6v9i/pdboutpath.mrc","/ziyingz/library/43cryo_Benchmark/43map/6v9i/6v9i.pdb",    True)
# input_map_path="/ziyingz/library/trainDiffmodeler/8h3r/traindata/input_diffmodeler_segment.mrc"
# final_pdb_output="/ziyingz/library/trainDiffmodeler/8h3r/8h3r.pdb"
# atom_label_path="/ziyingz/library/trainDiffmodeler/8h3r/label_closest_voxel.mrc"

# input_map_path="E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\8h3r\emd.map"
# final_pdb_output="E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\8h3r\8h3r.pdb"
# atom_label_path="E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\8h3r\label_closest_voxel.mrc"

# input_map_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\3j9p\emd.map"
# final_pdb_output = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\3j9p\3j9p.pdb"
# atom_label_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\3j9p\label_closest_voxel.mrc"

# 设置蛋白质名称
protein_name = "3j22"  # 修改此处的蛋白质名称，例如 "3j22" 或 "8h3r"

# 根据蛋白质名称自动设置路径
base_path = r"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA"
input_map_path = rf"E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\DATA\{protein_name}\processed\{protein_name}_segment.mrc"
final_pdb_output = os.path.join(base_path, protein_name, f"{protein_name}.pdb")
atom_label_path = os.path.join(base_path, protein_name, "label_closest_voxel.mrc")

# 调用函数
mask_map_by_pdb_slow(input_map_path, atom_label_path, final_pdb_output, cutoff=2, keep_label=True)

# pdb_info_dict = read_proteinpdb_data(final_pdb_output,run_type=1,atom_cutoff=2)
# #print(pdb_info_dict)
# map_data, mapc, mapr, maps, origin,nxstart,nystart,nzstart = process_map_data(input_map_path)
# distance_array,output_atom_label=assign_label_special(map_data, mapc, mapr, maps, origin,nxstart,nystart,nzstart,pdb_info_dict)

# print(distance_array)
# mask_map_by_pdb_slow(input_map_path,atom_label_path,final_pdb_output,cutoff=2,keep_label=True)
# save_label_map(input_map_path,atom_label_path,output_atom_label)