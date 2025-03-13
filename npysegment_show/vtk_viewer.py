#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VTK盒子查看器 - 用于在3D空间中可视化切割后的数据块
"""

import numpy as np
import vtk
from PyQt5.QtWidgets import QFrame
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VTKBoxViewer(QFrame):
    """使用VTK实现的3D盒子查看器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化VTK组件
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # 设置交互样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # 为选择盒子添加回调
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.interactor.SetPicker(self.picker)
        
        # 创建布局
        layout = QFrame().layout()
        layout = self.layout() if layout else None
        if layout is None:
            from PyQt5.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
        layout.addWidget(self.vtk_widget)
        
        # 数据存储
        self.mrc_data = None
        self.coord_data = None
        self.box_size = None
        self.npy_indices = None
        self.has_npy_boxes = {}  # 字典，存储哪些坐标有对应的NPY文件
        
        # 盒子演员字典，用于跟踪和交互
        self.box_actors = {}
        self.highlighted_actor = None
        
        # 回调函数
        self.box_selected_callback = None
        
        # 演员集合，用于管理所有的VTK演员
        self.actor_collection = vtk.vtkActorCollection()
    
    def initialize(self, mrc_data, coord_data, box_size, npy_indices):
        """初始化3D可视化"""
        self.mrc_data = mrc_data
        self.coord_data = coord_data
        self.box_size = box_size
        self.npy_indices = npy_indices
        
        # 创建有NPY文件的盒子的查找表
        self.has_npy_boxes = {}
        for idx, coord in enumerate(coord_data):
            key = tuple(coord)
            self.has_npy_boxes[key] = idx in npy_indices
        
        # 清除现有的演员
        self.renderer.RemoveAllViewProps()
        self.box_actors.clear()
        self.actor_collection.RemoveAllItems()
        
        # 创建整体MRC数据的轮廓
        self._create_mrc_outline()
        
        # 创建代表每个坐标的盒子
        self._create_boxes()
        
        # 设置回调函数，监听鼠标点击事件
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_click)
        
        # 设置相机位置
        self.renderer.ResetCamera()
        
        # 启动渲染
        self.renderer.GetActiveCamera().Elevation(30)
        self.renderer.GetActiveCamera().Azimuth(30)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.renderer.ResetCamera()
        self.interactor.Initialize()
        
        # 强制重绘
        self.vtk_widget.GetRenderWindow().Render()
    
    def _create_mrc_outline(self):
        """创建MRC数据体的轮廓"""
        outline = vtk.vtkOutlineFilter()
        
        # 创建MRC数据的体积数据集
        shape = self.mrc_data.shape
        volume_data = vtk.vtkImageData()
        volume_data.SetDimensions(shape[0], shape[1], shape[2])
        volume_data.SetSpacing(1, 1, 1)
        volume_data.SetOrigin(0, 0, 0)
        
        outline.SetInputData(volume_data)
        
        # 创建轮廓映射器和演员
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 灰色轮廓
        actor.GetProperty().SetLineWidth(1.0)
        
        # 添加到渲染器
        self.renderer.AddActor(actor)
        self.actor_collection.AddItem(actor)
    
    def _create_boxes(self):
        """为每个切割坐标创建盒子"""
        for idx, coord in enumerate(self.coord_data):
            x, y, z = coord
            
            # 判断这个坐标位置是否有对应的NPY文件
            has_npy = tuple(coord) in self.has_npy_boxes and self.has_npy_boxes[tuple(coord)]
            
            # 创建立方体
            box = vtk.vtkCubeSource()
            box.SetXLength(self.box_size)
            box.SetYLength(self.box_size)
            box.SetZLength(self.box_size)
            box.SetCenter(x + self.box_size/2, y + self.box_size/2, z + self.box_size/2)
            
            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(box.GetOutputPort())
            
            # 创建演员
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # 根据是否有NPY文件设置不同的颜色和透明度
            if has_npy:
                actor.GetProperty().SetColor(0.2, 0.8, 0.2)  # 绿色表示有NPY
                actor.GetProperty().SetOpacity(0.3)
            else:
                actor.GetProperty().SetColor(0.8, 0.2, 0.2)  # 红色表示没有NPY
                actor.GetProperty().SetOpacity(0.1)
            
            # 设置线框显示
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.0)
            
            # 添加到渲染器
            self.renderer.AddActor(actor)
            self.actor_collection.AddItem(actor)
            
            # 存储演员以供后续交互
            self.box_actors[idx] = {
                'actor': actor,
                'coord': coord,
                'has_npy': has_npy
            }
    
    def _on_click(self, obj, event):
        """处理鼠标点击事件"""
        # 获取点击位置
        click_pos = self.interactor.GetEventPosition()
        
        # 使用拾取器找出点击的演员
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        actor = self.picker.GetActor()
        
        if actor:
            # 在所有盒子中找到点击的那个
            for idx, box_info in self.box_actors.items():
                if box_info['actor'] == actor:
                    # 高亮显示选中的盒子
                    self._highlight_box(idx)
                    
                    # 调用回调函数
                    if self.box_selected_callback:
                        self.box_selected_callback(idx, box_info['coord'])
                    break
    
    def _highlight_box(self, box_idx):
        """高亮显示选中的盒子"""
        # 恢复之前高亮的盒子的状态
        if self.highlighted_actor:
            box_info = self.box_actors.get(self.highlighted_actor)
            if box_info:
                if box_info['has_npy']:
                    box_info['actor'].GetProperty().SetColor(0.2, 0.8, 0.2)
                    box_info['actor'].GetProperty().SetOpacity(0.3)
                else:
                    box_info['actor'].GetProperty().SetColor(0.8, 0.2, 0.2)
                    box_info['actor'].GetProperty().SetOpacity(0.1)
                box_info['actor'].GetProperty().SetRepresentationToWireframe()
                box_info['actor'].GetProperty().SetLineWidth(1.0)
        
        # 高亮新选中的盒子
        box_info = self.box_actors.get(box_idx)
        if box_info:
            box_info['actor'].GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色高亮
            box_info['actor'].GetProperty().SetOpacity(0.7)
            box_info['actor'].GetProperty().SetRepresentationToSurface()
            box_info['actor'].GetProperty().SetLineWidth(2.0)
            
            # 更新当前高亮的演员
            self.highlighted_actor = box_idx
            
            # 强制重绘
            self.vtk_widget.GetRenderWindow().Render()
    
    def highlight_box_by_coord(self, coord):
        """根据坐标高亮显示盒子"""
        for idx, box_info in self.box_actors.items():
            if np.array_equal(box_info['coord'], coord):
                self._highlight_box(idx)
                return True
        return False
    
    def set_box_selected_callback(self, callback):
        """设置盒子选中时的回调函数"""
        self.box_selected_callback = callback
    
    def clear(self):
        """清除所有可视化内容"""
        self.renderer.RemoveAllViewProps()
        self.box_actors.clear()
        self.actor_collection.RemoveAllItems()
        self.vtk_widget.GetRenderWindow().Render()
    
    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)
        self.vtk_widget.GetRenderWindow().Render()
