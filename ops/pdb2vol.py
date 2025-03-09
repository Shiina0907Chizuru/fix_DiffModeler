# pdb2vol.py adopted from BioTEMPy

from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
import os

import mrcfile
from scipy.ndimage import fourier_gaussian, gaussian_filter, zoom
from scipy.fftpack import fftn, ifftn

from tqdm import tqdm
from numba import njit
from numba.typed import Dict, List

# Dictionary of atom types and their corresponding masses.
# atom_mass_dict = {
#     "H": 1.008,
#     "C": 12.011,
#     "N": 14.007,
#     "O": 15.999,
#     "P": 30.974,  # for DNA/RNA
#     "S": 32.066,
# }

atom_mass_dict = Dict()

atom_mass_dict["H"] = 1.008
atom_mass_dict["C"] = 12.011
atom_mass_dict["CA"] = 12.011  # for PDB files without element notations
atom_mass_dict["N"] = 14.007
atom_mass_dict["O"] = 15.999
atom_mass_dict["P"] = 30.974
atom_mass_dict["S"] = 32.066


def get_atom_list(pdb_file, backbone_only=False):
    """
    Retrieve the coordinates and atom types from a PDB or CIF file.

    Parameters:
    pdb_file (str): The path to the PDB or CIF file.
    backbone_only (bool): If True, only backbone atoms will be extracted.

    Returns:
    tuple: A tuple containing two elements:
        - np.array: An array of atom coordinates.
        - list: A list of atom types.
        
    Raises:
    ValueError: If the file extension is not recognized or if the file cannot be parsed.
    """
    # 使用小写扩展名进行检查，确保大小写不敏感
    file_ext = os.path.splitext(pdb_file)[1].lower()
    
    try:
        # 首先尝试使用BioPython的解析器处理文件
        try:
            if file_ext == ".pdb":
                st_parser = PDBParser(QUIET=True)
                print(f"使用PDBParser解析文件: {pdb_file}")
            elif file_ext == ".cif":
                st_parser = MMCIFParser(QUIET=True)
                print(f"使用MMCIFParser解析文件: {pdb_file}")
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}，只支持.pdb和.cif格式")
                
            structure = st_parser.get_structure("protein", pdb_file)
            atom_list = []
            atom_type_list = List()

            if backbone_only:
                print("只提取骨架原子...")
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            try:
                                if "CA" in residue:
                                    atom_list.append(residue["CA"].get_coord())
                                    atom_type_list.append(residue["CA"].element)
                                    if "C" in residue:
                                        atom_list.append(residue["C"].get_coord())
                                        atom_type_list.append(residue["C"].element)
                                    if "N" in residue:
                                        atom_list.append(residue["N"].get_coord())
                                        atom_type_list.append(residue["N"].element)
                            except Exception as e:
                                print(f"处理残基时出错，跳过: {e}")
                                continue
            else:
                print("提取所有原子...")
                for atom in structure.get_atoms():
                    try:
                        atom_list.append(atom.get_coord())
                        atom_type_list.append(atom.element)
                    except Exception as e:
                        print(f"处理原子时出错，跳过: {e}")
                        continue
                    
            print(f"提取了{len(atom_list)}个原子")
            
            if len(atom_list) == 0:
                raise ValueError("未能从文件中提取任何原子")
            
            return np.array(atom_list), atom_type_list
            
        except Exception as primary_error:
            # BioPython解析失败，尝试备用方法
            print(f"BioPython解析失败: {primary_error}")
            print("尝试使用备用方法解析文件...")
            
            # 备用方法：直接解析PDB/CIF文件
            if file_ext == ".pdb":
                return parse_pdb_file_manually(pdb_file, backbone_only)
            elif file_ext == ".cif":
                return parse_cif_file_manually(pdb_file, backbone_only)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
        
    except Exception as e:
        print(f"解析文件{pdb_file}时出错: {e}")
        raise ValueError(f"无法解析{file_ext}文件: {e}")


def parse_pdb_file_manually(pdb_file, backbone_only=False):
    """手动解析PDB文件，提取原子坐标和类型"""
    print(f"尝试手动解析PDB文件: {pdb_file}")
    backbone_atoms = ["CA", "C", "N"]  # 只保留蛋白质主链原子
    # 移除RNA/DNA主链原子: P, O5', C5', C4', O4', C3', O3', C2', O2', C1'
    atom_list = []
    atom_type_list = List()
    
    try:
        # 检查文件是否存在
        if not os.path.exists(pdb_file):
            raise ValueError(f"文件不存在: {pdb_file}")
            
        # 检查文件大小
        if os.path.getsize(pdb_file) == 0:
            raise ValueError(f"文件为空: {pdb_file}")
        
        # 读取文件内容
        with open(pdb_file, 'r', errors='ignore') as f:
            lines = f.readlines()
            
        # 检查是否有ATOM行
        atom_lines = [line for line in lines if line.startswith("ATOM") or line.startswith("HETATM")]
        if not atom_lines:
            print(f"警告: 在PDB文件中没有找到ATOM或HETATM行: {pdb_file}")
            print("尝试使用第二种方法解析...")
            
            # 尝试更宽松的解析方法 - 查找包含原子信息的行
            atomic_data_lines = []
            for line in lines:
                # 查找可能包含原子数据的行 - 通常有3个浮点数(坐标)
                # 使用正则表达式查找包含数字和小数点的模式
                import re
                float_pattern = r'[-+]?\d*\.\d+'
                floats = re.findall(float_pattern, line)
                if len(floats) >= 3:  # 至少包含3个浮点数
                    atomic_data_lines.append(line)
            
            if not atomic_data_lines:
                raise ValueError(f"文件中未找到任何可能的原子坐标数据: {pdb_file}")
                
            # 从这些行中提取坐标
            for line in atomic_data_lines:
                try:
                    # 使用正则表达式查找浮点数
                    import re
                    floats = re.findall(r'[-+]?\d*\.\d+', line)
                    if len(floats) >= 3:
                        x = float(floats[0])
                        y = float(floats[1])
                        z = float(floats[2])
                        
                        # 尝试从行中提取元素类型
                        element = "C"  # 默认为碳
                        # 查找常见原子类型
                        for atom_type in ["CA", "C", "N", "O", "P", "S", "H"]:
                            if atom_type in line:
                                element = atom_type[0]
                                break
                                
                        atom_list.append([x, y, z])
                        atom_type_list.append(element)
                except Exception as e:
                    print(f"处理原子数据行时出错，跳过: {e}")
                    continue
        else:
            # 标准PDB文件解析
            for line in atom_lines:
                try:
                    # 确保行长度足够
                    if len(line) < 54:  # 最小长度，至少要包含坐标信息
                        continue
                        
                    # 更健壮地提取原子名称 - 处理格式异常的PDB文件
                    atom_name = ""
                    if len(line) >= 16:
                        try:
                            atom_name = line[12:16].strip()
                        except:
                            # 如果提取失败，尝试其他方法
                            parts = line.split()
                            if len(parts) > 2:
                                atom_name = parts[2]
                    
                    # 如果只提取骨架原子，则检查当前原子是否是骨架原子
                    if backbone_only and atom_name not in backbone_atoms:
                        continue
                    
                    # 提取坐标 - 使用更健壮的方法
                    x, y, z = None, None, None
                    
                    # 方法1: 标准PDB格式
                    try:
                        if len(line) >= 54:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                    except ValueError as ve:
                        print(f"标准方法解析坐标失败: {ve}")
                        # 特殊处理"数字+空格+浮点数"的情况，如"49 286.8"
                        try:
                            x_str = line[30:38].strip()
                            y_str = line[38:46].strip() 
                            z_str = line[46:54].strip()
                            
                            # 如果值包含空格，提取最后一个部分
                            if ' ' in x_str:
                                x = float(x_str.split()[-1])
                            else:
                                x = float(x_str)
                                
                            if ' ' in y_str:
                                y = float(y_str.split()[-1]) 
                            else:
                                y = float(y_str)
                                
                            if ' ' in z_str:
                                z = float(z_str.split()[-1])
                            else:
                                z = float(z_str)
                        except Exception as e:
                            print(f"处理特殊格式坐标失败: {e}")
                            pass
                        
                    # 方法2: 如果标准格式失败，尝试从行中提取任何浮点数
                    if x is None or y is None or z is None:
                        try:
                            import re
                            floats = re.findall(r'[-+]?\d*\.\d+', line)
                            if len(floats) >= 3:
                                x = float(floats[0])
                                y = float(floats[1])
                                z = float(floats[2])
                        except Exception as e:
                            print(f"提取浮点数失败: {e}")
                            pass
                            
                    # 如果仍然无法提取坐标，跳过这一行
                    if x is None or y is None or z is None:
                        print(f"无法从行中提取坐标，跳过: {line.strip()}")
                        continue
                            
                    # 提取元素类型
                    element = ""
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    
                    if not element:
                        # 如果元素字段为空，尝试从原子名称推断
                        if atom_name:
                            # 取第一个非数字字符
                            element = ''.join([c for c in atom_name if not c.isdigit()]).strip()[0:1]
                        else:
                            # 默认为碳原子
                            element = "C"
                    
                    atom_list.append([x, y, z])
                    atom_type_list.append(element)
                except Exception as e:
                    print(f"解析原子行时出错，跳过: {line.strip()}")
                    print(f"错误详情: {e}")
                    continue
    except Exception as e:
        print(f"读取或解析PDB文件时出错: {e}")
        # 尝试从文件名推断是否为PDB文件
        if not pdb_file.lower().endswith('.pdb'):
            print(f"警告: 文件可能不是有效的PDB文件: {pdb_file}")
        raise ValueError(f"无法解析PDB文件: {e}")
        
    if len(atom_list) == 0:
        # 如果没有提取到任何原子，尝试一种非常简单的方法 - 任何包含三个连续浮点数的行
        try:
            print("尝试最终的解析方法 - 搜索任何包含三个浮点数的行")
            with open(pdb_file, 'r', errors='ignore') as f:
                content = f.read()
                
            import re
            # 查找所有浮点数
            floats = re.findall(r'[-+]?\d*\.\d+', content)
            
            # 每三个浮点数作为一个原子的坐标
            for i in range(0, len(floats) - 2, 3):
                try:
                    x = float(floats[i])
                    y = float(floats[i+1])
                    z = float(floats[i+2])
                    atom_list.append([x, y, z])
                    atom_type_list.append("C")  # 默认为碳原子
                except Exception as e:
                    print(f"处理浮点数三元组失败: {e}")
                    continue
        except Exception as e:
            print(f"最终解析方法也失败: {e}")
            
    if len(atom_list) == 0:
        # 如果仍然没有提取到任何原子，则抛出错误
        raise ValueError(f"未能从PDB文件中提取任何原子: {pdb_file}")
        
    print(f"手动解析PDB文件提取了{len(atom_list)}个原子")
    return np.array(atom_list), atom_type_list


def parse_cif_file_manually(cif_file, backbone_only=False):
    """手动解析CIF文件，提取原子坐标和类型"""
    print(f"尝试手动解析CIF文件: {cif_file}")
    backbone_atoms = ["CA", "C", "N"]  # 只保留蛋白质主链原子
    # 移除RNA/DNA主链原子: P, O5', C5', C4', O4', C3', O3', C2', O2', C1'
    atom_list = []
    atom_type_list = List()
    
    try:
        with open(cif_file, 'r', errors='ignore') as f:
            cif_content = f.read()
            
        # 查找atom_site数据块
        import re
        atom_site_match = re.search(r'_atom_site\..*?(?=_|\Z)', cif_content, re.DOTALL)
        if not atom_site_match:
            raise ValueError("CIF文件中未找到atom_site数据块")
            
        atom_site_block = atom_site_match.group(0)
        
        # 提取列标题
        headers = re.findall(r'_atom_site\.(\S+)', atom_site_block)
        
        # 定位关键列
        col_indices = {}
        for key in ['group_PDB', 'label_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'type_symbol']:
            try:
                col_indices[key] = headers.index(key)
            except ValueError:
                col_indices[key] = -1
                
        # 确认必要的列都存在
        required_cols = ['Cartn_x', 'Cartn_y', 'Cartn_z']
        if any(col_indices[col] == -1 for col in required_cols):
            raise ValueError("CIF文件中缺少必要的坐标列")
            
        # 解析所有行
        data_lines = re.findall(r'\n(\S+(?:\s+\S+)*)', atom_site_block)
        for line in data_lines:
            try:
                parts = line.split()
                if len(parts) <= max(col_indices.values()):
                    continue
                    
                # 如果这是ATOM行
                if col_indices['group_PDB'] != -1 and parts[col_indices['group_PDB']] != 'ATOM':
                    continue
                    
                # 如果只提取骨架原子，检查原子名称
                if backbone_only:
                    if col_indices['label_atom_id'] == -1:
                        continue
                    atom_name = parts[col_indices['label_atom_id']].strip('"\'')
                    if atom_name not in backbone_atoms:
                        continue
                
                # 提取并处理坐标（处理可能存在的格式问题）
                try:
                    # 处理"数字+空格+浮点数"的格式问题
                    def parse_float_value(value_str):
                        # 如果值包含空格，可能是"数字+空格+浮点数"格式
                        if ' ' in value_str:
                            # 分割所有部分，取最后一个部分作为浮点数
                            parts = value_str.split()
                            return float(parts[-1])
                        else:
                            return float(value_str)
                    
                    # 提取X坐标
                    x_value = parts[col_indices['Cartn_x']]
                    x = parse_float_value(x_value)
                    
                    # 提取Y坐标
                    y_value = parts[col_indices['Cartn_y']]
                    y = parse_float_value(y_value)
                    
                    # 提取Z坐标
                    z_value = parts[col_indices['Cartn_z']]
                    z = parse_float_value(z_value)
                    
                except ValueError as ve:
                    print(f"无法解析坐标值: {ve}")
                    # 尝试替代方法：针对CIF中特殊格式进行处理
                    try:
                        # 尝试解析整行以确定坐标位置
                        # 在CIF文件中，通常坐标是连续的三个数值
                        found_coords = False
                        for i in range(len(parts) - 2):
                            try:
                                # 尝试连续的三个值作为坐标
                                x = float(parts[i].split()[-1] if ' ' in parts[i] else parts[i])
                                y = float(parts[i+1].split()[-1] if ' ' in parts[i+1] else parts[i+1])
                                z = float(parts[i+2].split()[-1] if ' ' in parts[i+2] else parts[i+2])
                                found_coords = True
                                break
                            except ValueError:
                                continue
                        
                        if not found_coords:
                            raise ValueError(f"无法在行中找到有效的坐标三元组: {line}")
                    except Exception as e2:
                        print(f"替代解析方法也失败: {e2}")
                        continue
                
                # 提取元素类型
                if col_indices['type_symbol'] != -1:
                    element = parts[col_indices['type_symbol']].strip('"\'')
                elif col_indices['label_atom_id'] != -1:
                    # 如果没有元素列，尝试从原子名称推断
                    atom_name = parts[col_indices['label_atom_id']].strip('"\'')
                    element = ''.join([c for c in atom_name if not c.isdigit()]).strip()[0:1]
                else:
                    # 默认为碳原子
                    element = "C"
                    
                atom_list.append([x, y, z])
                atom_type_list.append(element)
                
            except Exception as e:
                print(f"解析CIF数据行时出错，跳过: {e}")
                continue
                
    except Exception as e:
        print(f"读取CIF文件时出错: {e}")
        raise
        
    if len(atom_list) == 0:
        raise ValueError("未能从CIF文件中提取任何原子")
        
    print(f"手动解析CIF文件提取了{len(atom_list)}个原子")
    return np.array(atom_list), atom_type_list


def calculate_centre_of_mass(atom_list, atom_type_list):
    """
    Calculates the centre of mass for a given list of atoms and their types.

    Args:
        atom_list (np.ndarray): List of atom coordinates in the form (x, y, z).
        atom_type_list (list): List of atom types.

    Returns:
        tuple: The coordinates of the centre of mass in the form (x_co_m, y_co_m, z_co_m).
    """
    atom_list = np.array(atom_list)
    atom_type_list = np.array(atom_type_list)
    x = atom_list[:, 0]
    y = atom_list[:, 1]
    z = atom_list[:, 2]
    m = np.array([atom_mass_dict.get(atom_type, 0.0) for atom_type in atom_type_list])
    mass_total = np.sum(m)
    x_co_m = np.sum(x * m) / mass_total
    y_co_m = np.sum(y * m) / mass_total
    z_co_m = np.sum(z * m) / mass_total
    return x_co_m, y_co_m, z_co_m


def prot2map(
        atom_list,
        atom_type_list,
        voxel_size,
        resolution=None,
):
    """
    Calculate the size and origin of a protein map based on the given atom list,
    atom type list, voxel size, and optional resolution.

    Args:
        atom_list (numpy.ndarray): Array of atom coordinates.
        atom_type_list (list): Array of atom types.
        voxel_size (numpy.ndarray): The size of each voxel in x, y, and z dimensions.
        resolution (float, optional): Resolution of the map. Defaults to None.

    Returns:
        tuple: A tuple containing the size of the map in each dimension (z, y, x)
               and the origin of the map (x_origin, y_origin, z_origin).
    """
    max_x, max_y, max_z = atom_list.max(axis=0)
    min_x, min_y, min_z = atom_list.min(axis=0)

    if resolution is not None:
        edge = np.array(2 * resolution / voxel_size, dtype=int) + 4
    else:
        edge = np.array([10, 10, 10], dtype=int)

    x_size = int((max_x - min_x) / voxel_size[0]) + edge[0]
    y_size = int((max_y - min_y) / voxel_size[1]) + edge[1]
    z_size = int((max_z - min_z) / voxel_size[2]) + edge[2]

    CoM = calculate_centre_of_mass(atom_list, atom_type_list)

    # Origin calculated such that the centre of the map is the centre of
    # mass of the protein.
    half_x = max(CoM[0] - min_x, max_x - CoM[0])

    if half_x < (voxel_size[0] * x_size / 2.0):
        half_x = voxel_size[0] * x_size / 2.0
    x_origin = CoM[0] - half_x - edge[0] * voxel_size[0]
    x_size = int(half_x * 2.0 / voxel_size[0] + 2 * edge[0])
    half_y = max(CoM[1] - min_y, max_y - CoM[1])

    if half_y < (voxel_size[1] * y_size / 2.0):
        half_y = voxel_size[1] * y_size / 2.0
    y_origin = CoM[1] - half_y - edge[1] * voxel_size[1]
    y_size = int(half_y * 2.0 / voxel_size[1] + 2 * edge[1])
    half_z = max(CoM[2] - min_z, max_z - CoM[2])

    if half_z < (voxel_size[2] * z_size / 2.0):
        half_z = voxel_size[2] * z_size / 2.0
    z_origin = CoM[2] - half_z - edge[2] * voxel_size[2]
    z_size = int(half_z * 2.0 / voxel_size[2] + 2 * edge[2])

    return (z_size, y_size, x_size), (x_origin, y_origin, z_origin)


@njit(fastmath=True, nogil=True)
def mapGridPosition(origin, voxel_size, box_size, atom_coord):
    """
    Maps the coordinates of an atom to the corresponding grid position in a voxel grid.

    Parameters:
    origin (tuple): The origin of the voxel grid.
    voxel_size (tuple): The size of each voxel in the grid.
    box_size (tuple): The size of the voxel grid.
    atom_coord (tuple): The coordinates of the atom.

    Returns:
    tuple: The grid position (x, y, z) of the atom in the voxel grid.

    If the atom is outside the voxel grid, returns (0, 0, 0).
    """

    # NN interpolation
    # x_pos = int(round((atom_coord[0] - origin[0]) / voxel_size[0], 0))
    # y_pos = int(round((atom_coord[1] - origin[1]) / voxel_size[1], 0))
    # z_pos = int(round((atom_coord[2] - origin[2]) / voxel_size[2], 0))

    # No interpolation
    x_pos = (atom_coord[0] - origin[0]) / voxel_size[0]
    y_pos = (atom_coord[1] - origin[1]) / voxel_size[1]
    z_pos = (atom_coord[2] - origin[2]) / voxel_size[2]

    if (box_size[2] > x_pos + 1 >= 0) and (box_size[1] > y_pos + 1 >= 0) and (box_size[0] > z_pos + 1 >= 0):
        return x_pos, y_pos, z_pos
    else:
        return 0.0, 0.0, 0.0


@njit(fastmath=True, nogil=True)
def make_atom_overlay_map(origin, voxel_size, box_size, atom_list, atom_type_list, atom_mass_dict):
    """
    Creates an atom overlay map based on the given parameters.

    Parameters:
    origin (tuple): The origin coordinates of the map.
    voxel_size (float): The size of each voxel in the map.
    box_size (tuple): The size of the map in each dimension.
    atom_list (list): The list of atom coordinates.
    atom_type_list (list): The list of atom types.

    Returns:
    numpy.ndarray: The atom overlay map.
    """
    map_data = np.zeros(box_size)
    for atom, atom_type in zip(atom_list, atom_type_list):
        pos = mapGridPosition(origin, voxel_size, box_size, atom)
        if not (pos[0] == pos[1] == pos[2] == 0.0):
            atom_mass = atom_mass_dict.get(atom_type, 0.0)
            pos_x_0 = int(np.floor(pos[0]))
            pos_y_0 = int(np.floor(pos[1]))
            pos_z_0 = int(np.floor(pos[2]))
            pos_x_1 = pos_x_0 + 1
            pos_y_1 = pos_y_0 + 1
            pos_z_1 = pos_z_0 + 1

            a = pos_x_1 - pos[0]
            b = pos_y_1 - pos[1]
            c = pos_z_1 - pos[2]

            # Trilinear interpolation to surrounding vertices
            map_data[pos_z_0, pos_y_0, pos_x_0] += a * b * c * atom_mass
            map_data[pos_z_1, pos_y_0, pos_x_0] += a * b * (1 - c) * atom_mass
            map_data[pos_z_0, pos_y_1, pos_x_0] += a * (1 - b) * c * atom_mass
            map_data[pos_z_0, pos_y_0, pos_x_1] += (1 - a) * b * c * atom_mass
            map_data[pos_z_1, pos_y_1, pos_x_0] += a * (1 - b) * (1 - c) * atom_mass
            map_data[pos_z_0, pos_y_1, pos_x_1] += (1 - a) * (1 - b) * c * atom_mass
            map_data[pos_z_1, pos_y_0, pos_x_1] += (1 - a) * b * (1 - c) * atom_mass
            map_data[pos_z_1, pos_y_1, pos_x_1] += (1 - a) * (1 - b) * (1 - c) * atom_mass

    return map_data


def write_mrc_file(data, origin, voxel_size, mrc_file):
    """
    Write a data array to an MRC file.

    Args:
        data (ndarray): The data array to be written.
        origin (tuple): The origin coordinates of the data.
        voxel_size (array_like): The voxel size of the data.
        mrc_file (str): The path to the output MRC file.

    Returns:
        None
    """
    # 检查数据是否包含负值
    has_negative = np.any(data < 0)
    min_val = np.min(data)
    
    if has_negative:
        print(f"数据包含负值，最小值: {min_val:.6f}，确保MRC文件使用mode 2（浮点模式）")
    
    # 1. 确保数据是浮点类型
    float_data = data.astype(np.float32)
    
    # 2. 创建一个新的MRC文件，确保使用overwrite=True
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        # 3. 设置数据 
        mrc.set_data(float_data)
        
        # 4. 显式设置模式为2（32位浮点）
        mrc.header.mode = 2
        
        # 5. 设置体素大小和原点
        mrc.voxel_size = tuple(voxel_size)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]
        
        # 6. 确保更新头部信息
        mrc.update_header_stats()
    
    # 验证文件是否正确保存
    try:
        with mrcfile.open(mrc_file) as mrc:
            saved_min = np.min(mrc.data)
            saved_max = np.max(mrc.data)
            
            if has_negative and saved_min >= 0:
                print(f"警告: MRC文件可能未成功保存负值。文件中最小值为 {saved_min:.6f}，但原始数据最小值为 {min_val:.6f}")
            else:
                print(f"MRC文件已成功保存，数据范围: {saved_min:.6f} 到 {saved_max:.6f}")
                print(f"MRC文件模式: {mrc.header.mode}")
    except Exception as e:
        print(f"验证MRC文件时出错: {e}")


def blur_map(data, resolution, sigma_coeff):
    """
    Blurs the input data using a Gaussian filter.

    Args:
        data (ndarray): The input data to be blurred.
        resolution (float): The resolution of the data.
        sigma_coeff (float): The coefficient to determine the sigma value for the Gaussian filter.

    Returns:
        ndarray: The blurred data.

    """
    sigma = resolution * sigma_coeff
    new_data = fourier_gaussian(fftn(data), sigma)
    return np.real(ifftn(new_data))


def blur_map_real_space(data, resolution, sigma_coeff):
    """
    Blurs a map in real space using a Gaussian filter.

    Args:
        data (ndarray): The input map data.
        resolution (float): The resolution of the map.
        sigma_coeff (float): The coefficient to determine the sigma value for the Gaussian filter.

    Returns:
        ndarray: The blurred map data.
    """
    sigma = resolution * sigma_coeff
    print(sigma)
    new_data = gaussian_filter(data, sigma)
    return new_data


def normalize_map(map_data):
    """
    Normalize a map by subtracting the mean and dividing by the standard deviation.

    Parameters:
    map_data (numpy.ndarray): The input map data.

    Returns:
    numpy.ndarray: The normalized map data, with mean=0 and standard deviation=1.
    """
    if map_data.std() != 0:
        # 标准Z-score归一化：减去均值，除以标准差
        return (map_data - map_data.mean()) / map_data.std()
    else:
        return map_data


def resample_by_box_size(data, box_size):
    """
    Resamples the given data array to match the specified box size using cubic spline interpolation.

    Parameters:
        data (ndarray): The input data array.
        box_size (tuple): The desired box size for resampling.

    Returns:
        ndarray: The resampled data array.
    """
    # cubic spline interpolation
    zoom_factor = np.array(box_size) / np.array(data.shape)
    return zoom(data, zoom_factor, order=3)


def pdb2vol(
        input_pdb,
        resolution,
        output_mrc=None,
        ref_map=None,
        sigma_coeff=0.356,
        real_space=False,
        normalize=True,
        backbone_only=False,
        contour=False,
        apply_contour=True,
        bin_mask=False,
        return_data=False,
):
    """
    Convert a PDB or CIF file to a volumetric map in MRC format.

    Args:
        input_pdb (str): Path to the input PDB or CIF file.
        output_mrc (str): Path to save the output MRC file.
        resolution (float): Resolution of the output map.
        ref_map (str, optional): Path to a reference map in MRC format. Defaults to None.
        sigma_coeff (float, optional): Sigma coefficient for blurring. Defaults to 0.356.
        real_space (bool, optional): Whether to perform real-space blurring. Defaults to False.
        normalize (bool, optional): Whether to normalize the output map. Defaults to True.
        backbone_only (bool, optional): Whether to use only backbone atoms. Defaults to False.
        contour (float, optional): Contour level for thresholding. Defaults to False.
        apply_contour (bool, optional): Whether to apply contour thresholding. Defaults to True.
        bin_mask (bool, optional): Whether to binarize the output map. Defaults to False.
        return_data (bool, optional): Whether to return the map data. Defaults to False.

    Raises:
        ValueError: If the input file is not a PDB or CIF file.
        ValueError: If no atoms are found in the input file.
        ValueError: If the number of atoms and atom types do not match.

    Returns:
        None or ndarray: Returns the map data if return_data is True, otherwise None.
    """

    # sigma_coeff = 1/(pi*sqrt(2*log(2)) = 0.187, makes the FT fall to half maximum at wavenumber 1/resolution
    # sigma_coeff = 1/(pi*sqrt(2)) = 0.225 is the default value used in Chimera, makes the Fourier transform (FT) of the distribution fall to 1/e of its maximum value at wavenumber 1/resolution
    # sigma_coeff = 1/(2*sqrt(2)) =  0.356 makes the Gaussian width at 1/e maximum height equal the resolution
    # sigma_coeff = 1/(2*sqrt(2*log(2))) = 0.4247 makes the Gaussian width at half maximum height equal the resolution

    # 检查文件扩展名，改为大小写不敏感
    file_ext = os.path.splitext(input_pdb)[1].lower()
    if file_ext not in [".pdb", ".cif"]:
        raise ValueError(f"输入文件必须是PDB或CIF格式。当前文件扩展名: {file_ext}")
    
    print(f"处理输入文件: {input_pdb}，文件类型: {'CIF' if file_ext == '.cif' else 'PDB'}")
    
    try:
        atoms, types = get_atom_list(input_pdb, backbone_only=backbone_only)
    except Exception as e:
        raise ValueError(f"无法解析{file_ext}文件: {e}")

    if len(atoms) == 0:
        raise ValueError("No atoms found in input file")
    if len(atoms) != len(types):
        raise ValueError("Number of atoms and atom types does not match")

    if not ref_map:
        r = np.clip(resolution / 4.0, a_min=1.0, a_max=3.5)
        voxel_size = np.array([r, r, r])
        dims, origin = prot2map(atoms, types, voxel_size, resolution)
    else:
        print(f"使用参考密度图: {ref_map}")
        try:
            with mrcfile.open(ref_map, permissive=True) as mrc:
                voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
                dims = mrc.data.shape
                origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z])
                print(f"参考密度图尺寸: {dims}")
                print(f"参考密度图体素大小: {voxel_size}")
                print(f"参考密度图原点: {origin}")
        except Exception as e:
            print(f"读取参考密度图失败: {e}")
            print("将使用自动计算的尺寸和原点")
            r = np.clip(resolution / 4.0, a_min=1.0, a_max=3.5)
            voxel_size = np.array([r, r, r])
            dims, origin = prot2map(atoms, types, voxel_size, resolution)

    x_s = int(dims[2] * voxel_size[2])
    y_s = int(dims[1] * voxel_size[1])
    z_s = int(dims[0] * voxel_size[0])

    new_voxel_size = np.array(
        [voxel_size[2] * dims[2] / x_s, voxel_size[1] * dims[1] / y_s, voxel_size[0] * dims[0] / z_s])

    # print(new_voxel_size)

    map_data = make_atom_overlay_map(origin, new_voxel_size, (z_s, y_s, x_s), atoms, types, atom_mass_dict)

    if resolution * sigma_coeff / new_voxel_size[0] >= 1:
        if real_space:
            blurred_data = blur_map_real_space(map_data, resolution, sigma_coeff)
        else:
            blurred_data = blur_map(map_data, resolution, sigma_coeff)
    else:
        print("Warning: Blurring will not be performed because the resolution is too high w.r.t. the voxel size.")
        blurred_data = map_data  # no blurring

    blurred_data = resample_by_box_size(blurred_data, dims)

    # # 输出归一化前的blurred_data统计信息
    # min_val = np.min(blurred_data)
    # max_val = np.max(blurred_data)
    # mean_val = np.mean(blurred_data)
    # std_val = np.std(blurred_data)
    # print(f"归一化前的blurred_data统计信息: 最小值={min_val:.6f}, 最大值={max_val:.6f}, 均值={mean_val:.6f}, 标准差={std_val:.6f}")

    if normalize:
        blurred_data = normalize_map(blurred_data)
        # 输出归一化后的统计信息
        min_val = np.min(blurred_data)
        max_val = np.max(blurred_data)
        mean_val = np.mean(blurred_data)
        std_val = np.std(blurred_data)
        print(f"归一化后的blurred_data统计信息: 最小值={min_val:.6f}, 最大值={max_val:.6f}, 均值={mean_val:.6f}, 标准差={std_val:.6f}")
        
        # 计算正负值比例
        positive_pct = np.sum(blurred_data > 0) / blurred_data.size * 100
        negative_pct = np.sum(blurred_data < 0) / blurred_data.size * 100
        print(f"正值占比: {positive_pct:.2f}%, 负值占比: {negative_pct:.2f}%")

    # 记录contour和bin_mask处理前的状态
    print(f"在contour和bin_mask处理前，数据包含负值: {np.any(blurred_data < 0)}")
    print(f"contour参数值: {contour}, apply_contour: {apply_contour}")
    print(f"bin_mask参数值: {bin_mask}")

    if contour and apply_contour:
        print(f"执行contour处理，contour阈值: {contour}")
        blurred_data = np.where(blurred_data > contour, blurred_data, 0)
        print(f"contour处理后，数据是否包含负值: {np.any(blurred_data < 0)}")

    if bin_mask:
        print("执行bin_mask操作")
        # binarize to get a mask
        blurred_data = np.where(blurred_data > 0, 1, 0)
        print(f"bin_mask处理后，数据是否包含负值: {np.any(blurred_data < 0)}")

    # 写入MRC文件前记录最终状态
    if output_mrc is not None:
        print(f"写入MRC文件前，数据是否包含负值: {np.any(blurred_data < 0)}")
        print(f"写入MRC文件前，最小值: {np.min(blurred_data):.6f}")
        write_mrc_file(blurred_data, origin, voxel_size, output_mrc)

    if return_data:
        return blurred_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a PDB or CIF file to a volumetric map in MRC format.")
    parser.add_argument("input_pdb", help="Path to the input PDB or CIF file.")
    parser.add_argument("resolution", type=float, help="Resolution of the output map.")
    parser.add_argument("output_mrc", help="Path to save the output MRC file.")
    parser.add_argument("-m", "--ref_map", help="Path to a reference map in MRC format.", default=None)
    parser.add_argument("-s", "--sigma_coeff", type=float, default=0.356, help="Sigma coefficient for blurring.")
    parser.add_argument("-r", "--real_space", action="store_true", default=False,
                        help="Whether to perform real-space blurring.")
    parser.add_argument("-n", "--normalize", action="store_true", default=True,
                        help="Whether to normalize the output map.")
    parser.add_argument("-bb", "--backbone_only", action="store_true", default=False,
                        help="Whether to only consider backbone atoms.")
    parser.add_argument("-b", "--bin_mask", action="store_true", default=False,
                        help="Whether to binarize the output map.")
    parser.add_argument("-c", "--contour", type=float, default=0.0, help="Contour level for contouring the output map.")
    parser.add_argument("-ac", "--apply_contour", action="store_true", default=True,
                        help="Whether to apply contour thresholding.")
    args = parser.parse_args()

    pdb2vol(
        args.input_pdb,
        args.resolution,
        args.output_mrc,
        args.ref_map,
        args.sigma_coeff,
        args.real_space,
        args.normalize,
        args.backbone_only,
        args.contour,
        args.apply_contour,
        args.bin_mask,
        False,
    )
