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
    QLabel, QGridLayout, QSplitter, QLineEdit, QMessageBox,
    QTabWidget, QCheckBox, QGroupBox, QRadioButton, QButtonGroup
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
        self.resize(1600, 1000)
        
        # 初始化数据管理器
        self.data_manager = DataManager()
        
        # 初始化当前选中的盒子坐标
        self.current_selected_coord = None
        self.current_selected_view = 0  # 0=输入视图, 1=输出视图
        
        # 添加相机同步锁，防止循环调用
        self.camera_sync_lock = False
        
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
        
        # 输入MRC文件加载
        load_layout.addWidget(QLabel("输入MRC文件:"), 0, 0)
        self.mrc_path_label = QLabel("未选择")
        load_layout.addWidget(self.mrc_path_label, 0, 1)
        self.load_mrc_btn = QPushButton("加载输入MRC")
        self.load_mrc_btn.clicked.connect(self.load_mrc_file)
        load_layout.addWidget(self.load_mrc_btn, 0, 2)
        
        # 输出MRC文件加载
        load_layout.addWidget(QLabel("输出MRC文件:"), 1, 0)
        self.output_mrc_path_label = QLabel("未选择")
        load_layout.addWidget(self.output_mrc_path_label, 1, 1)
        self.load_output_mrc_btn = QPushButton("加载输出MRC")
        self.load_output_mrc_btn.clicked.connect(self.load_output_mrc_file)
        load_layout.addWidget(self.load_output_mrc_btn, 1, 2)
        
        # Coord文件加载
        load_layout.addWidget(QLabel("坐标文件:"), 2, 0)
        self.coord_path_label = QLabel("未选择")
        load_layout.addWidget(self.coord_path_label, 2, 1)
        self.load_coord_btn = QPushButton("加载坐标")
        self.load_coord_btn.clicked.connect(self.load_coord_file)
        load_layout.addWidget(self.load_coord_btn, 2, 2)
        
        # Input NPY目录加载
        load_layout.addWidget(QLabel("输入NPY目录:"), 3, 0)
        self.input_dir_label = QLabel("未选择")
        load_layout.addWidget(self.input_dir_label, 3, 1)
        self.load_input_btn = QPushButton("加载输入NPY")
        self.load_input_btn.clicked.connect(self.load_input_dir)
        load_layout.addWidget(self.load_input_btn, 3, 2)
        
        # Output NPY目录加载
        load_layout.addWidget(QLabel("输出NPY目录:"), 4, 0)
        self.output_dir_label = QLabel("未选择")
        load_layout.addWidget(self.output_dir_label, 4, 1)
        self.load_output_btn = QPushButton("加载输出NPY")
        self.load_output_btn.clicked.connect(self.load_output_dir)
        load_layout.addWidget(self.load_output_btn, 4, 2)
        
        # 初始化可视化按钮
        self.visualize_btn = QPushButton("初始化可视化")
        self.visualize_btn.clicked.connect(self.initialize_visualization)
        self.visualize_btn.setEnabled(False)
        load_layout.addWidget(self.visualize_btn, 5, 0, 1, 3)
        
        left_layout.addWidget(load_group)
        
        # 添加可视化控制选项
        control_group = QGroupBox("可视化控制")
        control_layout = QVBoxLayout(control_group)
        
        # 边界显示控制
        self.boundary_check = QCheckBox("显示有数据区域边界")
        self.boundary_check.setEnabled(False)
        self.boundary_check.stateChanged.connect(self.toggle_boundary_display)
        control_layout.addWidget(self.boundary_check)
        
        # 同步控制选项
        self.sync_views_check = QCheckBox("同步两个视图")
        self.sync_views_check.setChecked(True)
        self.sync_views_check.setEnabled(False)
        control_layout.addWidget(self.sync_views_check)
        
        left_layout.addWidget(control_group)
        
        # 添加坐标查找部分
        find_group = QGroupBox("坐标查找")
        find_layout = QHBoxLayout(find_group)
        
        find_layout.addWidget(QLabel("查找盒子:"))
        self.coord_x_input = QLineEdit()
        self.coord_x_input.setPlaceholderText("X")
        find_layout.addWidget(self.coord_x_input)
        
        self.coord_y_input = QLineEdit()
        self.coord_y_input.setPlaceholderText("Y")
        find_layout.addWidget(self.coord_y_input)
        
        self.coord_z_input = QLineEdit()
        self.coord_z_input.setPlaceholderText("Z")
        find_layout.addWidget(self.coord_z_input)
        
        self.find_box_btn = QPushButton("查找")
        self.find_box_btn.clicked.connect(self.find_box_by_coord)
        self.find_box_btn.setEnabled(False)
        find_layout.addWidget(self.find_box_btn)
        
        left_layout.addWidget(find_group)
        
        # 添加导出MRC功能
        export_group = QGroupBox("导出MRC")
        export_layout = QVBoxLayout(export_group)
        
        # 导出当前选中的盒子
        export_current_layout = QHBoxLayout()
        self.export_current_box_btn = QPushButton("导出当前选中的盒子")
        self.export_current_box_btn.clicked.connect(self.export_current_box)
        self.export_current_box_btn.setEnabled(False)
        export_current_layout.addWidget(self.export_current_box_btn)
        
        # 选择导出类型
        self.export_type_input = QRadioButton("输入数据")
        self.export_type_input.setChecked(True)
        self.export_type_output = QRadioButton("输出数据")
        export_type_group = QButtonGroup(export_group)
        export_type_group.addButton(self.export_type_input)
        export_type_group.addButton(self.export_type_output)
        
        export_type_layout = QHBoxLayout()
        export_type_layout.addWidget(QLabel("导出数据类型:"))
        export_type_layout.addWidget(self.export_type_input)
        export_type_layout.addWidget(self.export_type_output)
        
        # 导出所有盒子
        export_all_layout = QHBoxLayout()
        self.export_all_boxes_btn = QPushButton("导出所有有数据的盒子")
        self.export_all_boxes_btn.clicked.connect(self.export_all_boxes)
        self.export_all_boxes_btn.setEnabled(False)
        export_all_layout.addWidget(self.export_all_boxes_btn)
        
        export_layout.addLayout(export_type_layout)
        export_layout.addLayout(export_current_layout)
        export_layout.addLayout(export_all_layout)
        
        left_layout.addWidget(export_group)
        
        # 添加NPY信息列表
        self.npy_list = QListWidget()
        self.npy_list.setMinimumWidth(300)
        self.npy_list.itemClicked.connect(self.on_npy_item_selected)
        left_layout.addWidget(QLabel("NPY文件:"))
        left_layout.addWidget(self.npy_list)
        
        # 创建右侧VTK可视化区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 创建标签页用于显示不同的视图
        self.view_tabs = QTabWidget()
        self.view_tabs.currentChanged.connect(self.on_tab_changed)
        
        # 创建输入视图
        self.vtk_viewer_input = VTKBoxViewer(view_id=0)
        
        # 创建输出视图
        self.vtk_viewer_output = VTKBoxViewer(view_id=1)
        
        # 为双视图对比创建独立的VTK查看器
        self.vtk_viewer_input_compare = VTKBoxViewer(view_id=0)
        self.vtk_viewer_output_compare = VTKBoxViewer(view_id=1)
        
        # 输入和输出视图对比标签页
        compare_tab = QWidget()
        compare_layout = QHBoxLayout(compare_tab)
        
        # 创建分割器以便更好地控制两个视图的大小
        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.addWidget(self.vtk_viewer_input_compare)
        view_splitter.addWidget(self.vtk_viewer_output_compare)
        view_splitter.setSizes([500, 500])  # 设置初始分割比例为相等
        
        # 将分割器添加到比较布局中
        compare_layout.addWidget(view_splitter)
        
        # 添加对比视图标签页
        self.view_tabs.addTab(compare_tab, "双视图对比")
        
        # 添加单独的输入视图标签页
        input_tab = QWidget()
        input_layout = QHBoxLayout(input_tab)
        input_layout.addWidget(self.vtk_viewer_input)
        self.view_tabs.addTab(input_tab, "输入视图")
        
        # 添加单独的输出视图标签页
        output_tab = QWidget()
        output_layout = QHBoxLayout(output_tab)
        output_layout.addWidget(self.vtk_viewer_output)
        self.view_tabs.addTab(output_tab, "输出视图")
        
        right_layout.addWidget(self.view_tabs)
        
        # 将左右两侧加入主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1200])  # 初始分割比例
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
    def load_mrc_file(self):
        """加载MRC文件"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择MRC文件", "", "MRC Files (*.mrc *.map);;All Files (*)")
        if filepath:
            self.data_manager.load_mrc_file(filepath)
            self.mrc_path_label.setText(os.path.basename(filepath))
            self._check_if_ready()
    
    def load_output_mrc_file(self):
        """加载输出MRC文件"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择输出MRC文件", "", "MRC Files (*.mrc *.map);;All Files (*)")
        if filepath:
            self.data_manager.load_output_mrc_file(filepath)
            self.output_mrc_path_label.setText(os.path.basename(filepath))
            self._check_if_ready()
    
    def load_coord_file(self):
        """加载坐标文件"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择坐标文件", "", "NPY Files (*.npy);;All Files (*)")
        if filepath:
            self.data_manager.load_coord_file(filepath)
            self.coord_path_label.setText(os.path.basename(filepath))
            self._check_if_ready()
    
    def load_input_dir(self):
        """加载输入NPY目录"""
        dirpath = QFileDialog.getExistingDirectory(self, "选择输入NPY目录")
        if dirpath:
            self.data_manager.load_input_dir(dirpath)
            self.input_dir_label.setText(os.path.basename(dirpath))
            self._update_npy_list()
            self._check_if_ready()
    
    def load_output_dir(self):
        """加载输出NPY目录"""
        dirpath = QFileDialog.getExistingDirectory(self, "选择输出NPY目录")
        if dirpath:
            self.data_manager.load_output_dir(dirpath)
            self.output_dir_label.setText(os.path.basename(dirpath))
            self._update_npy_list()
            self._check_if_ready()
    
    def _check_if_ready(self):
        """检查是否已经加载了所有必要的数据"""
        if self.data_manager.mrc_data is not None and self.data_manager.coord_data is not None:
            self.visualize_btn.setEnabled(True)
        else:
            self.visualize_btn.setEnabled(False)
    
    def _update_npy_list(self):
        """更新NPY文件列表"""
        self.npy_list.clear()
        
        for npy_type, npy_idx, coord in self.data_manager.get_npy_list():
            # 添加到列表中，格式为: Input 42 [x, y, z]
            item_text = f"{npy_type.capitalize()} {npy_idx} [{coord[0]}, {coord[1]}, {coord[2]}]"
            self.npy_list.addItem(item_text)
    
    def initialize_visualization(self):
        """初始化3D可视化"""
        try:
            # 初始化输入视图
            self.vtk_viewer_input.initialize(
                self.data_manager.mrc_data,
                self.data_manager.coord_data,
                self.data_manager.box_size,
                self.data_manager.get_npy_indices()
            )
            
            # 初始化比较视图中的输入视图
            self.vtk_viewer_input_compare.initialize(
                self.data_manager.mrc_data,
                self.data_manager.coord_data,
                self.data_manager.box_size,
                self.data_manager.get_npy_indices()
            )
            
            # 如果有输出MRC数据，则初始化输出视图
            if self.data_manager.output_mrc_data is not None:
                mrc_data = self.data_manager.output_mrc_data
            else:
                # 如果没有输出MRC数据，则使用输入MRC数据
                mrc_data = self.data_manager.mrc_data
                
            # 初始化输出视图
            self.vtk_viewer_output.initialize(
                mrc_data,
                self.data_manager.coord_data,
                self.data_manager.box_size,
                self.data_manager.get_npy_indices()
            )
            
            # 初始化比较视图中的输出视图
            self.vtk_viewer_output_compare.initialize(
                mrc_data,
                self.data_manager.coord_data,
                self.data_manager.box_size,
                self.data_manager.get_npy_indices()
            )
            
            # 启用坐标查找功能
            self.find_box_btn.setEnabled(True)
            
            # 启用导出MRC功能
            self.export_current_box_btn.setEnabled(True)
            self.export_all_boxes_btn.setEnabled(True)
            
            # 启用边界显示控制
            self.boundary_check.setEnabled(True)
            
            # 启用同步控制选项
            self.sync_views_check.setEnabled(True)
            
            # 设置回调函数，当在3D视图中选择了盒子时
            self.vtk_viewer_input.set_box_selected_callback(self.on_box_selected_from_3d)
            self.vtk_viewer_output.set_box_selected_callback(self.on_box_selected_from_3d)
            self.vtk_viewer_input_compare.set_box_selected_callback(self.on_box_selected_from_3d)
            self.vtk_viewer_output_compare.set_box_selected_callback(self.on_box_selected_from_3d)
            
            # 设置相机变化回调，用于同步视图
            self.vtk_viewer_input.set_camera_changed_callback(self.on_camera_changed)
            self.vtk_viewer_output.set_camera_changed_callback(self.on_camera_changed)
            self.vtk_viewer_input_compare.set_camera_changed_callback(self.on_camera_changed)
            self.vtk_viewer_output_compare.set_camera_changed_callback(self.on_camera_changed)
            
            # 显示坐标轴
            self.vtk_viewer_input.show_axes()
            self.vtk_viewer_output.show_axes()
            self.vtk_viewer_input_compare.show_axes()
            self.vtk_viewer_output_compare.show_axes()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化可视化失败: {str(e)}")
    
    def on_npy_item_selected(self, item):
        """当从NPY列表中选择一个项目时"""
        # 解析项目文本
        text = item.text()
        parts = text.split()
        
        npy_type = parts[0].lower()  # Input 或 Output
        npy_idx = int(parts[1])
        
        # 提取坐标
        coord_text = text[text.find("[")+1:text.find("]")]
        coord = [int(x.strip()) for x in coord_text.split(",")]
        
        self.current_selected_coord = coord
        
        # 在所有3D视图中高亮显示对应的盒子
        self.vtk_viewer_input.highlight_box_by_coord(coord)
        self.vtk_viewer_output.highlight_box_by_coord(coord)
        self.vtk_viewer_input_compare.highlight_box_by_coord(coord)
        self.vtk_viewer_output_compare.highlight_box_by_coord(coord)
        
        # 显示对应的NPY数据
        self.data_manager.show_npy_data(npy_type, npy_idx)
    
    def find_box_by_coord(self):
        """根据输入的坐标查找盒子"""
        try:
            x = int(self.coord_x_input.text())
            y = int(self.coord_y_input.text())
            z = int(self.coord_z_input.text())
            
            coord = [x, y, z]
            self.current_selected_coord = coord
            
            # 在所有3D视图中高亮显示对应的盒子
            found_input = self.vtk_viewer_input.highlight_box_by_coord(coord)
            found_output = self.vtk_viewer_output.highlight_box_by_coord(coord)
            self.vtk_viewer_input_compare.highlight_box_by_coord(coord)
            self.vtk_viewer_output_compare.highlight_box_by_coord(coord)
            
            if not (found_input or found_output):
                QMessageBox.warning(self, "未找到", "在指定坐标找不到盒子。")
                return
            
            # 查找该坐标是否有对应的NPY文件
            npy_info = self.data_manager.find_npy_by_coord(coord)
            if npy_info:
                npy_type, npy_idx = npy_info
                
                # 显示对应的NPY数据
                self.data_manager.show_npy_data(npy_type, npy_idx)
            
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的坐标。")
    
    def on_box_selected_from_3d(self, view_id, coord, has_npy):
        """当从3D视图中选择了盒子时的回调函数"""
        if coord is None:
            return
        
        self.current_selected_coord = coord
        self.current_selected_view = view_id
        
        # 更新坐标输入框
        self.coord_x_input.setText(str(coord[0]))
        self.coord_y_input.setText(str(coord[1]))
        self.coord_z_input.setText(str(coord[2]))
        
        # 在所有视图中高亮显示选中的盒子
        if self.sync_views_check.isChecked():
            # 确保在所有四个视图中都高亮显示
            self.vtk_viewer_input.highlight_box_by_coord(coord)
            self.vtk_viewer_output.highlight_box_by_coord(coord)
            self.vtk_viewer_input_compare.highlight_box_by_coord(coord)
            self.vtk_viewer_output_compare.highlight_box_by_coord(coord)
        
        # 查找该坐标是否有对应的NPY文件
        npy_info = self.data_manager.find_npy_by_coord(coord)
        if npy_info:
            npy_type, npy_idx = npy_info
            
            # 显示对应的NPY数据
            self.data_manager.show_npy_data(npy_type, npy_idx)
    
    def on_camera_changed(self, view_id, camera_position, focal_point, view_up):
        """当相机位置变化时的回调函数，用于同步两个视图"""
        if not self.sync_views_check.isChecked() or self.camera_sync_lock:
            return
        
        # 设置锁以防止循环调用
        self.camera_sync_lock = True
        
        try:
            # 根据发生变化的视图ID，同步到其他所有视图
            if view_id == 0:  # 输入视图变化
                # 同步到所有其他视图
                self.vtk_viewer_output.set_camera_position(camera_position, focal_point, view_up)
                self.vtk_viewer_output_compare.set_camera_position(camera_position, focal_point, view_up)
                self.vtk_viewer_input_compare.set_camera_position(camera_position, focal_point, view_up)
            
            elif view_id == 1:  # 输出视图变化
                # 同步到所有其他视图
                self.vtk_viewer_input.set_camera_position(camera_position, focal_point, view_up)
                self.vtk_viewer_input_compare.set_camera_position(camera_position, focal_point, view_up)
                self.vtk_viewer_output_compare.set_camera_position(camera_position, focal_point, view_up)
        finally:
            # 释放锁
            self.camera_sync_lock = False
    
    def toggle_boundary_display(self):
        """切换边界显示"""
        if self.boundary_check.isChecked():
            # 在所有视图中显示边界
            self.vtk_viewer_input.show_boundary()
            self.vtk_viewer_output.show_boundary()
            self.vtk_viewer_input_compare.show_boundary()
            self.vtk_viewer_output_compare.show_boundary()
        else:
            # 在所有视图中隐藏边界
            self.vtk_viewer_input.hide_boundary()
            self.vtk_viewer_output.hide_boundary()
            self.vtk_viewer_input_compare.hide_boundary()
            self.vtk_viewer_output_compare.hide_boundary()
    
    def export_current_box(self):
        """导出当前选中的盒子为MRC文件"""
        if self.current_selected_coord is None:
            QMessageBox.warning(self, "未选择盒子", "请先选择一个盒子。")
            return
        
        # 确定导出类型
        if self.export_type_input.isChecked():
            mrc_type = "input"
            if self.data_manager.mrc_data is None:
                QMessageBox.warning(self, "缺少数据", "未加载输入MRC数据。")
                return
        else:
            mrc_type = "output"
            if self.data_manager.output_mrc_data is None:
                QMessageBox.warning(self, "缺少数据", "未加载输出MRC数据。")
                return
        
        # 选择保存文件
        filepath, _ = QFileDialog.getSaveFileName(
            self, 
            "导出MRC文件", 
            f"{mrc_type}_box_x{self.current_selected_coord[0]}_y{self.current_selected_coord[1]}_z{self.current_selected_coord[2]}.mrc", 
            "MRC Files (*.mrc);;All Files (*)"
        )
        
        if filepath:
            success = self.data_manager.export_box_as_mrc(
                self.current_selected_coord, 
                filepath, 
                mrc_type
            )
            
            if success:
                QMessageBox.information(self, "导出成功", f"已成功导出MRC文件到: {filepath}")
            else:
                QMessageBox.critical(self, "导出失败", "导出MRC文件时发生错误。")
    
    def export_all_boxes(self):
        """导出所有有NPY数据的盒子为MRC文件"""
        # 确定导出类型
        if self.export_type_input.isChecked():
            mrc_type = "input"
            if self.data_manager.mrc_data is None:
                QMessageBox.warning(self, "缺少数据", "未加载输入MRC数据。")
                return
        else:
            mrc_type = "output"
            if self.data_manager.output_mrc_data is None:
                QMessageBox.warning(self, "缺少数据", "未加载输出MRC数据。")
                return
        
        # 选择保存目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return
        
        # 导出所有盒子
        success_count = self.data_manager.export_all_boxes_as_mrc(export_dir, mrc_type)
        
        if success_count > 0:
            QMessageBox.information(
                self, 
                "导出成功", 
                f"已成功导出 {success_count} 个MRC文件到: {export_dir}"
            )
        else:
            QMessageBox.warning(self, "导出警告", "没有成功导出任何MRC文件。")
    
    def on_tab_changed(self, index):
        """当用户切换标签页时触发"""
        # 强制刷新VTK渲染窗口
        if index == 0:  # 双视图对比
            if hasattr(self, 'vtk_viewer_input_compare') and hasattr(self, 'vtk_viewer_output_compare'):
                self.vtk_viewer_input_compare.update()
                self.vtk_viewer_output_compare.update()
        elif index == 1:  # 输入视图
            if hasattr(self, 'vtk_viewer_input'):
                self.vtk_viewer_input.update()
        elif index == 2:  # 输出视图
            if hasattr(self, 'vtk_viewer_output'):
                self.vtk_viewer_output.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
