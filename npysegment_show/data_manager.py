#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理器 - 用于加载和管理MRC文件、坐标数据和NPY文件
"""

import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shutil

class DataManager:
    """管理可视化工具所需的所有数据"""
    
    def __init__(self):
        # 初始化数据存储
        self.mrc_data = None      # 输入MRC数据
        self.mrc_path = None      # 输入MRC路径
        self.output_mrc_data = None  # 输出MRC数据
        self.output_mrc_path = None  # 输出MRC路径
        self.coord_data = None
        self.coord_path = None
        self.input_dir = None
        self.output_dir = None
        self.input_files = {}  # 存储input npy文件的字典 {idx: filepath}
        self.output_files = {}  # 存储output npy文件的字典 {idx: filepath}
        
        # 基于generate_input_data.py的参数
        self.box_size = 64  # 默认盒子大小
        self.stride = 32    # 默认步幅
        
        # 可视化用的matplotlib图像
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # MRC文件头信息
        self.mrc_header = None
        self.output_mrc_header = None
    
    def load_mrc_file(self, filepath):
        """加载MRC文件并存储数据"""
        self.mrc_path = filepath
        
        # 使用mrcfile库加载MRC数据
        with mrcfile.open(filepath, permissive=True) as mrc:
            self.mrc_data = np.array(mrc.data)
            self.mrc_header = mrc.header
            
        print(f"Loaded MRC file: {filepath}")
        print(f"MRC data shape: {self.mrc_data.shape}")
        print(f"MRC data range: {np.min(self.mrc_data)} to {np.max(self.mrc_data)}")
        
        return self.mrc_data
    
    def load_output_mrc_file(self, filepath):
        """加载输出MRC文件并存储数据"""
        self.output_mrc_path = filepath
        
        # 使用mrcfile库加载MRC数据
        with mrcfile.open(filepath, permissive=True) as mrc:
            self.output_mrc_data = np.array(mrc.data)
            self.output_mrc_header = mrc.header
            
        print(f"Loaded Output MRC file: {filepath}")
        print(f"Output MRC data shape: {self.output_mrc_data.shape}")
        print(f"Output MRC data range: {np.min(self.output_mrc_data)} to {np.max(self.output_mrc_data)}")
        
        return self.output_mrc_data
    
    def load_coord_file(self, filepath):
        """加载坐标文件并存储数据"""
        self.coord_path = filepath
        self.coord_data = np.load(filepath)
        
        print(f"Loaded Coord file: {filepath}")
        print(f"Number of coordinates: {len(self.coord_data)}")
        
        return self.coord_data
    
    def load_input_dir(self, dirpath):
        """加载包含输入NPY文件的目录"""
        self.input_dir = dirpath
        self.input_files = {}
        
        # 识别所有input_*.npy文件并建立索引映射
        for filename in os.listdir(dirpath):
            if filename.startswith("input_") and filename.endswith(".npy"):
                # 从文件名提取索引
                try:
                    idx = int(filename.replace("input_", "").replace(".npy", ""))
                    self.input_files[idx] = os.path.join(dirpath, filename)
                except ValueError:
                    print(f"Skipping file with invalid format: {filename}")
        
        print(f"Loaded Input directory: {dirpath}")
        print(f"Number of input NPY files: {len(self.input_files)}")
        
        return self.input_files
    
    def load_output_dir(self, dirpath):
        """加载包含输出NPY文件的目录"""
        self.output_dir = dirpath
        self.output_files = {}
        
        # 识别所有output_*.npy文件并建立索引映射
        for filename in os.listdir(dirpath):
            if filename.startswith("output_") and filename.endswith(".npy"):
                # 从文件名提取索引
                try:
                    idx = int(filename.replace("output_", "").replace(".npy", ""))
                    self.output_files[idx] = os.path.join(dirpath, filename)
                except ValueError:
                    print(f"Skipping file with invalid format: {filename}")
        
        print(f"Loaded Output directory: {dirpath}")
        print(f"Number of output NPY files: {len(self.output_files)}")
        
        return self.output_files
    
    def get_npy_indices(self):
        """获取所有有NPY文件的坐标索引"""
        indices = []
        
        # 如果coord_data还没有加载，返回空列表
        if self.coord_data is None:
            return []
        
        # 检查每个坐标是否有对应的input NPY文件
        for idx in range(len(self.coord_data)):
            if idx in self.input_files:
                indices.append(idx)
        
        return indices
    
    def get_npy_list(self):
        """获取所有NPY文件的信息列表，用于显示"""
        npy_list = []
        
        # 如果coord_data还没有加载，返回空列表
        if self.coord_data is None:
            return []
        
        # 添加所有input NPY文件
        for idx, filepath in self.input_files.items():
            if idx < len(self.coord_data):
                npy_list.append(("input", idx, self.coord_data[idx]))
        
        # 添加所有output NPY文件
        for idx, filepath in self.output_files.items():
            if idx < len(self.coord_data):
                npy_list.append(("output", idx, self.coord_data[idx]))
        
        return npy_list
    
    def show_npy_data(self, npy_type, npy_idx):
        """显示指定的NPY数据"""
        # 获取文件路径
        if npy_type == "input" and npy_idx in self.input_files:
            filepath = self.input_files[npy_idx]
        elif npy_type == "output" and npy_idx in self.output_files:
            filepath = self.output_files[npy_idx]
        else:
            print(f"No {npy_type} NPY file found with index {npy_idx}")
            return
        
        # 加载NPY数据
        try:
            npy_data = np.load(filepath)
            self._visualize_npy_data(npy_data, npy_type, npy_idx)
        except Exception as e:
            print(f"Failed to load NPY data: {str(e)}")
    
    def _visualize_npy_data(self, npy_data, npy_type, npy_idx):
        """使用matplotlib可视化NPY数据"""
        # 清除现有的图表
        self.fig.clear()
        
        # 获取NPY数据的中心切片
        center_slice = npy_data.shape[0] // 2
        
        # 创建子图
        axs = self.fig.subplots(2, 2)
        
        # 绘制XY平面的切片
        axs[0, 0].imshow(npy_data[center_slice, :, :], cmap='viridis')
        axs[0, 0].set_title(f'XY Plane (Z={center_slice})')
        axs[0, 0].set_xlabel('Y')
        axs[0, 0].set_ylabel('X')
        
        # 绘制XZ平面的切片
        axs[0, 1].imshow(npy_data[:, center_slice, :], cmap='viridis')
        axs[0, 1].set_title(f'XZ Plane (Y={center_slice})')
        axs[0, 1].set_xlabel('Z')
        axs[0, 1].set_ylabel('X')
        
        # 绘制YZ平面的切片
        axs[1, 0].imshow(npy_data[:, :, center_slice], cmap='viridis')
        axs[1, 0].set_title(f'YZ Plane (X={center_slice})')
        axs[1, 0].set_xlabel('Z')
        axs[1, 0].set_ylabel('Y')
        
        # 绘制3D体积投影
        # 简单的最大强度投影
        projection = np.max(npy_data, axis=0)
        axs[1, 1].imshow(projection, cmap='viridis')
        axs[1, 1].set_title('Maximum Intensity Projection')
        axs[1, 1].set_xlabel('Z')
        axs[1, 1].set_ylabel('Y')
        
        # 添加文本信息
        if npy_type == "input":
            title = f"Input NPY {npy_idx}"
        else:
            title = f"Output NPY {npy_idx}"
            
        if npy_idx < len(self.coord_data):
            coord = self.coord_data[npy_idx]
            title += f" at [{coord[0]}, {coord[1]}, {coord[2]}]"
        
        self.fig.suptitle(title, fontsize=14)
        
        # 调整子图之间的距离
        self.fig.tight_layout()
        
        # 更新画布
        self.canvas.draw()
        
        # 显示统计信息
        print(f"\nStatistics for {title}:")
        print(f"Shape: {npy_data.shape}")
        print(f"Data range: {np.min(npy_data):.6f} to {np.max(npy_data):.6f}")
        print(f"Mean: {np.mean(npy_data):.6f}, Std: {np.std(npy_data):.6f}")
        
        # 分析有意义密度的比例（基于generate_input_data.py中的阈值判断）
        meaningful_density_count = np.sum(npy_data > 0)
        meaningful_density_ratio = meaningful_density_count / npy_data.size
        print(f"Meaningful density ratio: {meaningful_density_ratio:.6f} ({meaningful_density_count} voxels)")
        
        # 检查负值
        neg_count = np.sum(npy_data < 0)
        if neg_count > 0:
            neg_ratio = neg_count / npy_data.size
            print(f"Contains {neg_count} negative values ({neg_ratio:.6%} of total)")
    
    def find_npy_by_coord(self, coord):
        """根据坐标查找对应的NPY文件"""
        # 在坐标列表中找到匹配的索引
        for idx, c in enumerate(self.coord_data):
            if np.array_equal(c, coord):
                # 检查是否有对应的input NPY文件
                if idx in self.input_files:
                    return ("input", idx)
                # 检查是否有对应的output NPY文件
                elif idx in self.output_files:
                    return ("output", idx)
        
        return None
    
    def export_box_as_mrc(self, coord, output_filepath, mrc_type="input"):
        """
        将指定坐标的盒子导出为MRC文件
        
        参数:
        coord - 盒子坐标 [x, y, z]
        output_filepath - 输出MRC文件路径
        mrc_type - "input" 或 "output" 表示使用哪个MRC数据源
        
        返回:
        bool - 是否成功导出
        """
        try:
            # 选择数据源
            if mrc_type == "input":
                if self.mrc_data is None:
                    print("No input MRC data loaded")
                    return False
                src_data = self.mrc_data
                header = self.mrc_header
            else:
                if self.output_mrc_data is None:
                    print("No output MRC data loaded")
                    return False
                src_data = self.output_mrc_data
                header = self.output_mrc_header
            
            # 提取盒子数据
            x, y, z = coord
            x_end = min(x + self.box_size, src_data.shape[0])
            y_end = min(y + self.box_size, src_data.shape[1])
            z_end = min(z + self.box_size, src_data.shape[2])
            
            box_data = np.zeros((self.box_size, self.box_size, self.box_size), dtype=src_data.dtype)
            
            # 复制数据
            box_data[:x_end-x, :y_end-y, :z_end-z] = src_data[x:x_end, y:y_end, z:z_end]
            
            # 创建新的MRC文件
            with mrcfile.new(output_filepath, overwrite=True) as mrc:
                mrc.set_data(box_data)
                
                # 复制原始MRC的相关头信息
                if header is not None:
                    # 设置新的原点，使其与原始MRC相对应
                    origin_x = header.origin.x + x if hasattr(header.origin, 'x') else 0
                    origin_y = header.origin.y + y if hasattr(header.origin, 'y') else 0
                    origin_z = header.origin.z + z if hasattr(header.origin, 'z') else 0
                    
                    # 设置像素大小
                    mrc.voxel_size.x = header.voxel_size.x if hasattr(header, 'voxel_size') and hasattr(header.voxel_size, 'x') else 1.0
                    mrc.voxel_size.y = header.voxel_size.y if hasattr(header, 'voxel_size') and hasattr(header.voxel_size, 'y') else 1.0
                    mrc.voxel_size.z = header.voxel_size.z if hasattr(header, 'voxel_size') and hasattr(header.voxel_size, 'z') else 1.0
                    
                    # 设置原点
                    mrc.header.origin.x = origin_x
                    mrc.header.origin.y = origin_y
                    mrc.header.origin.z = origin_z
            
            print(f"Successfully exported MRC box to {output_filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to export MRC box: {str(e)}")
            return False
    
    def export_all_boxes_as_mrc(self, output_dir, mrc_type="input"):
        """
        导出坐标文件中所有有NPY对应的盒子为MRC文件
        
        参数:
        output_dir - 输出目录路径
        mrc_type - "input" 或 "output" 表示使用哪个MRC数据源
        
        返回:
        int - 成功导出的盒子数量
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        success_count = 0
        total_count = 0
        
        # 获取需要导出的坐标
        coord_indices = self.get_npy_indices()
        
        for idx in coord_indices:
            if idx < len(self.coord_data):
                coord = self.coord_data[idx]
                output_filepath = os.path.join(output_dir, f"{mrc_type}_box_{idx}_x{coord[0]}_y{coord[1]}_z{coord[2]}.mrc")
                
                if self.export_box_as_mrc(coord, output_filepath, mrc_type):
                    success_count += 1
                
                total_count += 1
        
        print(f"Exported {success_count} of {total_count} MRC boxes to {output_dir}")
        return success_count
