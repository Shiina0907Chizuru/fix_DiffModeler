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
    
    def __init__(self, parent=None, view_id=0):
        super().__init__(parent)
        
        # 视图ID (0=输入视图, 1=输出视图)
        self.view_id = view_id
        
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
        
        # 边界显示标志
        self.show_boundaries = False
        self.boundary_actor = None
        
        # 坐标轴演员
        self.axes_actor = None
        self.axes_text_actors = []
        
        # 回调函数
        self.box_selected_callback = None
        
        # 同步视图的回调
        self.camera_changed_callback = None
        
        # 演员集合，用于管理所有的VTK演员
        self.actor_collection = vtk.vtkActorCollection()
        
        # 相机位置观察器
        self.camera_observer = None
    
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
        
        # 创建坐标轴
        self._create_coordinate_axes()
        
        # 设置回调函数，监听鼠标点击事件
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_click)
        
        # 监听相机变化事件
        camera = self.renderer.GetActiveCamera()
        if self.camera_observer:
            camera.RemoveObserver(self.camera_observer)
        self.camera_observer = camera.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_camera_changed)
        
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
    
    def _create_coordinate_axes(self):
        """创建坐标轴"""
        # 移除之前的坐标轴
        if self.axes_actor:
            self.renderer.RemoveActor(self.axes_actor)
        
        for text_actor in self.axes_text_actors:
            self.renderer.RemoveActor(text_actor)
        self.axes_text_actors = []
        
        # 创建原点和坐标轴
        axes = vtk.vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        axes.SetTotalLength(self.box_size*2, self.box_size*2, self.box_size*2)
        
        # 放置在左下角
        shape = self.mrc_data.shape
        origin_x, origin_y, origin_z = 0, 0, 0
        
        # 创建坐标轴变换
        transform = vtk.vtkTransform()
        transform.Translate(origin_x, origin_y, origin_z)
        axes.SetUserTransform(transform)
        
        # 添加到渲染器
        self.renderer.AddActor(axes)
        self.axes_actor = axes
        
        # 添加原点文本
        origin_text = vtk.vtkTextActor()
        origin_text.SetInput(f"Origin: ({origin_x}, {origin_y}, {origin_z})")
        origin_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # 白色
        origin_text.GetTextProperty().SetFontSize(12)
        origin_text.SetPosition(10, 10)
        self.renderer.AddActor2D(origin_text)
        self.axes_text_actors.append(origin_text)
        
        # 强制重绘
        self.vtk_widget.GetRenderWindow().Render()
    
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
                        # 更新参数顺序：视图ID, 坐标, 是否有NPY数据
                        self.box_selected_callback(self.view_id, box_info['coord'], box_info['has_npy'])
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
    
    def set_camera_changed_callback(self, callback):
        """设置相机变化时的回调函数"""
        self.camera_changed_callback = callback
    
    def _on_camera_changed(self, obj, event):
        """处理相机变化事件"""
        if self.camera_changed_callback:
            camera = self.renderer.GetActiveCamera()
            position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
            view_up = camera.GetViewUp()
            # 更新参数顺序：视图ID, 位置, 焦点, 向上向量
            self.camera_changed_callback(self.view_id, position, focal_point, view_up)
    
    def update_camera(self, position, focal_point, view_up):
        """从另一个视图更新相机位置"""
        # 避免递归调用
        if self.camera_observer:
            camera = self.renderer.GetActiveCamera()
            camera.RemoveObserver(self.camera_observer)
            
        # 更新相机位置
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)
        
        # 重新添加观察器
        self.camera_observer = camera.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_camera_changed)
        
        # 更新渲染
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_camera_position(self, position, focal_point, view_up):
        """设置相机位置和方向"""
        self.update_camera(position, focal_point, view_up)
    
    def show_boundary(self):
        """显示有NPY数据的区域边界"""
        self.toggle_boundary_display(True)
    
    def hide_boundary(self):
        """隐藏有NPY数据的区域边界"""
        self.toggle_boundary_display(False)
    
    def show_axes(self):
        """显示坐标轴"""
        if self.axes_actor:
            self.axes_actor.VisibilityOn()
            for text_actor in self.axes_text_actors:
                text_actor.VisibilityOn()
            self.vtk_widget.GetRenderWindow().Render()
    
    def hide_axes(self):
        """隐藏坐标轴"""
        if self.axes_actor:
            self.axes_actor.VisibilityOff()
            for text_actor in self.axes_text_actors:
                text_actor.VisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_boundary_display(self, show_boundaries):
        """切换边界显示"""
        self.show_boundaries = show_boundaries
        
        # 移除现有的边界演员
        if self.boundary_actor:
            self.renderer.RemoveActor(self.boundary_actor)
            self.boundary_actor = None
        
        if show_boundaries:
            # 创建有数据的区域的集合
            data_points = vtk.vtkPoints()
            for idx, box_info in self.box_actors.items():
                if box_info['has_npy']:
                    x, y, z = box_info['coord']
                    # 添加盒子的8个顶点
                    data_points.InsertNextPoint(x, y, z)
                    data_points.InsertNextPoint(x + self.box_size, y, z)
                    data_points.InsertNextPoint(x, y + self.box_size, z)
                    data_points.InsertNextPoint(x + self.box_size, y + self.box_size, z)
                    data_points.InsertNextPoint(x, y, z + self.box_size)
                    data_points.InsertNextPoint(x + self.box_size, y, z + self.box_size)
                    data_points.InsertNextPoint(x, y + self.box_size, z + self.box_size)
                    data_points.InsertNextPoint(x + self.box_size, y + self.box_size, z + self.box_size)
            
            # 创建多边形数据
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(data_points)
            
            # 创建包围盒
            delaunay3D = vtk.vtkDelaunay3D()
            delaunay3D.SetInputData(polydata)
            delaunay3D.Update()
            
            # 提取表面
            surface = vtk.vtkGeometryFilter()
            surface.SetInputConnection(delaunay3D.GetOutputPort())
            surface.Update()
            
            # 创建映射器和演员
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(surface.GetOutputPort())
            
            self.boundary_actor = vtk.vtkActor()
            self.boundary_actor.SetMapper(mapper)
            self.boundary_actor.GetProperty().SetColor(0.0, 0.8, 0.8)  # 青色边界
            self.boundary_actor.GetProperty().SetOpacity(0.5)
            self.boundary_actor.GetProperty().SetRepresentationToWireframe()
            self.boundary_actor.GetProperty().SetLineWidth(3.0)
            
            # 添加到渲染器
            self.renderer.AddActor(self.boundary_actor)
        
        # 强制重绘
        self.vtk_widget.GetRenderWindow().Render()
    
    def clear(self):
        """清除所有可视化内容"""
        self.renderer.RemoveAllViewProps()
        self.box_actors.clear()
        self.actor_collection.RemoveAllItems()
        self.highlighted_actor = None
        self.boundary_actor = None
        self.axes_actor = None
        self.axes_text_actors = []
        self.vtk_widget.GetRenderWindow().Render()
    
    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)
        if hasattr(self, 'renderer') and hasattr(self, 'vtk_widget') and self.renderer and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()
    
    def showEvent(self, event):
        """处理显示事件"""
        super().showEvent(event)
        if hasattr(self, 'renderer') and hasattr(self, 'vtk_widget') and self.renderer and self.vtk_widget:
            # 强制重绘窗口
            self.vtk_widget.GetRenderWindow().Render()
            
    def hideEvent(self, event):
        """处理隐藏事件"""
        super().hideEvent(event)
        
    def moveEvent(self, event):
        """处理移动事件"""
        super().moveEvent(event)
        if hasattr(self, 'renderer') and hasattr(self, 'vtk_widget') and self.renderer and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()
            
    def enterEvent(self, event):
        """处理鼠标进入事件"""
        super().enterEvent(event)
        if hasattr(self, 'renderer') and hasattr(self, 'vtk_widget') and self.renderer and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()
