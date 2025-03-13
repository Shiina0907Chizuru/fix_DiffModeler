#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NPY Segment Visualizer - DiffModeler
这个工具用于可视化DiffModeler处理过程中生成的NPY切片数据
"""

import sys
import os
import numpy as np
import mrcfile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QFileDialog, QListWidget, 
    QLabel, QGridLayout, QSplitter, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QSize

# 导入自定义模块
from vtk_viewer import VTKBoxViewer
from data_manager import DataManager

class MainWindow(QMainWindow):
    """NPY分段可视化器的主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和尺寸
        self.setWindowTitle("DiffModeler NPY Segment Visualizer")
        self.resize(1200, 800)
        
        # 初始化数据管理器
        self.data_manager = DataManager()
        
        # 创建UI
        self.create_ui()
        
    def create_ui(self):
        """创建用户界面"""
        # 创建主窗口部件和布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # 创建左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 添加数据加载部分
        load_group = QWidget()
        load_layout = QGridLayout(load_group)
        
        # MRC文件加载
        load_layout.addWidget(QLabel("MRC File:"), 0, 0)
        self.mrc_path_label = QLabel("Not selected")
        load_layout.addWidget(self.mrc_path_label, 0, 1)
        self.load_mrc_btn = QPushButton("Load MRC")
        self.load_mrc_btn.clicked.connect(self.load_mrc_file)
        load_layout.addWidget(self.load_mrc_btn, 0, 2)
        
        # Coord文件加载
        load_layout.addWidget(QLabel("Coord File:"), 1, 0)
        self.coord_path_label = QLabel("Not selected")
        load_layout.addWidget(self.coord_path_label, 1, 1)
        self.load_coord_btn = QPushButton("Load Coord")
        self.load_coord_btn.clicked.connect(self.load_coord_file)
        load_layout.addWidget(self.load_coord_btn, 1, 2)
        
        # Input NPY目录加载
        load_layout.addWidget(QLabel("Input NPY Dir:"), 2, 0)
        self.input_dir_label = QLabel("Not selected")
        load_layout.addWidget(self.input_dir_label, 2, 1)
        self.load_input_btn = QPushButton("Load Input")
        self.load_input_btn.clicked.connect(self.load_input_dir)
        load_layout.addWidget(self.load_input_btn, 2, 2)
        
        # Output NPY目录加载
        load_layout.addWidget(QLabel("Output NPY Dir:"), 3, 0)
        self.output_dir_label = QLabel("Not selected")
        load_layout.addWidget(self.output_dir_label, 3, 1)
        self.load_output_btn = QPushButton("Load Output")
        self.load_output_btn.clicked.connect(self.load_output_dir)
        load_layout.addWidget(self.load_output_btn, 3, 2)
        
        # 初始化可视化按钮
        self.visualize_btn = QPushButton("Initialize Visualization")
        self.visualize_btn.clicked.connect(self.initialize_visualization)
        self.visualize_btn.setEnabled(False)
        load_layout.addWidget(self.visualize_btn, 4, 0, 1, 3)
        
        left_layout.addWidget(load_group)
        
        # 添加坐标查找部分
        find_group = QWidget()
        find_layout = QHBoxLayout(find_group)
        
        find_layout.addWidget(QLabel("Find Box:"))
        self.coord_x_input = QLineEdit()
        self.coord_x_input.setPlaceholderText("X")
        find_layout.addWidget(self.coord_x_input)
        
        self.coord_y_input = QLineEdit()
        self.coord_y_input.setPlaceholderText("Y")
        find_layout.addWidget(self.coord_y_input)
        
        self.coord_z_input = QLineEdit()
        self.coord_z_input.setPlaceholderText("Z")
        find_layout.addWidget(self.coord_z_input)
        
        self.find_box_btn = QPushButton("Find")
        self.find_box_btn.clicked.connect(self.find_box_by_coord)
        self.find_box_btn.setEnabled(False)
        find_layout.addWidget(self.find_box_btn)
        
        left_layout.addWidget(find_group)
        
        # 添加NPY信息列表
        self.npy_list = QListWidget()
        self.npy_list.setMinimumWidth(300)
        self.npy_list.itemClicked.connect(self.on_npy_item_selected)
        left_layout.addWidget(QLabel("NPY Files:"))
        left_layout.addWidget(self.npy_list)
        
        # 创建右侧VTK可视化区域
        self.vtk_viewer = VTKBoxViewer()
        
        # 将左右两侧加入主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.vtk_viewer)
        splitter.setSizes([300, 900])  # 初始分割比例
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
    def load_mrc_file(self):
        """加载MRC文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MRC File", "", "MRC Files (*.mrc *.map);;All Files (*)"
        )
        if file_path:
            try:
                self.data_manager.load_mrc_file(file_path)
                self.mrc_path_label.setText(os.path.basename(file_path))
                self.check_initialize_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load MRC file: {str(e)}")
    
    def load_coord_file(self):
        """加载坐标文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Coord File", "", "NPY Files (*.npy);;All Files (*)"
        )
        if file_path:
            try:
                self.data_manager.load_coord_file(file_path)
                self.coord_path_label.setText(os.path.basename(file_path))
                self.check_initialize_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Coord file: {str(e)}")
    
    def load_input_dir(self):
        """加载Input NPY目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Input NPY Directory"
        )
        if dir_path:
            try:
                self.data_manager.load_input_dir(dir_path)
                self.input_dir_label.setText(os.path.basename(dir_path))
                self.check_initialize_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Input directory: {str(e)}")
    
    def load_output_dir(self):
        """加载Output NPY目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output NPY Directory"
        )
        if dir_path:
            try:
                self.data_manager.load_output_dir(dir_path)
                self.output_dir_label.setText(os.path.basename(dir_path))
                self.check_initialize_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Output directory: {str(e)}")
    
    def check_initialize_ready(self):
        """检查是否准备好初始化可视化"""
        # MRC和Coord文件都加载后才能初始化
        if self.data_manager.mrc_data is not None and self.data_manager.coord_data is not None:
            self.visualize_btn.setEnabled(True)
    
    def initialize_visualization(self):
        """初始化3D可视化"""
        try:
            # 初始化VTK查看器
            self.vtk_viewer.initialize(
                self.data_manager.mrc_data,
                self.data_manager.coord_data,
                self.data_manager.box_size,
                self.data_manager.get_npy_indices()
            )
            
            # 更新NPY列表
            self.update_npy_list()
            
            # 启用坐标查找功能
            self.find_box_btn.setEnabled(True)
            
            # 设置回调函数，当在3D视图中选择了盒子时
            self.vtk_viewer.set_box_selected_callback(self.on_box_selected_from_3d)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize visualization: {str(e)}")
    
    def update_npy_list(self):
        """更新NPY文件列表"""
        self.npy_list.clear()
        npy_files = self.data_manager.get_npy_list()
        
        for idx, (npy_type, npy_idx, coord) in enumerate(npy_files):
            if npy_type == "input":
                prefix = "Input"
            else:
                prefix = "Output"
            
            self.npy_list.addItem(f"{prefix} {npy_idx}: [{coord[0]}, {coord[1]}, {coord[2]}]")
    
    def on_npy_item_selected(self, item):
        """NPY列表项被选中时"""
        # 解析选中的项以获取NPY类型和索引
        text = item.text()
        parts = text.split(":")
        npy_info = parts[0].strip()
        coord_text = parts[1].strip()[1:-1]  # 去除方括号
        coord = [int(x.strip()) for x in coord_text.split(",")]
        
        if npy_info.startswith("Input"):
            npy_type = "input"
            npy_idx = int(npy_info.split()[1])
        else:
            npy_type = "output"
            npy_idx = int(npy_info.split()[1])
        
        # 在3D视图中高亮显示对应的盒子
        self.vtk_viewer.highlight_box_by_coord(coord)
        
        # 显示对应的NPY数据
        self.data_manager.show_npy_data(npy_type, npy_idx)
    
    def find_box_by_coord(self):
        """根据输入的坐标查找盒子"""
        try:
            x = int(self.coord_x_input.text())
            y = int(self.coord_y_input.text())
            z = int(self.coord_z_input.text())
            
            # 在3D视图中高亮显示对应的盒子
            found = self.vtk_viewer.highlight_box_by_coord([x, y, z])
            
            if not found:
                QMessageBox.warning(self, "Not Found", "No box found at the specified coordinates.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integer coordinates.")
    
    def on_box_selected_from_3d(self, coord_idx, coord):
        """当从3D视图中选择了盒子时的回调"""
        # 查找对应的NPY文件
        npy_item = self.data_manager.find_npy_by_coord(coord)
        
        if npy_item is not None:
            npy_type, npy_idx = npy_item
            
            # 在列表中找到并选中对应的项
            for i in range(self.npy_list.count()):
                item = self.npy_list.item(i)
                if (npy_type == "input" and f"Input {npy_idx}:" in item.text()) or \
                   (npy_type == "output" and f"Output {npy_idx}:" in item.text()):
                    self.npy_list.setCurrentItem(item)
                    break
            
            # 显示对应的NPY数据
            self.data_manager.show_npy_data(npy_type, npy_idx)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
