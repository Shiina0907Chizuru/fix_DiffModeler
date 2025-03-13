# NPY Segment Visualizer for DiffModeler

这个工具用于可视化DiffModeler处理过程中生成的NPY切片数据，帮助研究人员更好地理解和分析模型的输入和输出。

## 功能特点

- 加载并显示原始MRC文件
- 加载坐标文件(Coord.npy)，显示切割区域
- 显示输入和输出NPY文件的3D盒子表示
- 交互式选择和查看特定盒子的NPY数据
- 通过坐标查找特定盒子
- 区分有NPY数据和无NPY数据的区域（基于generate_input_data.py中的筛选逻辑）
- 支持旋转、缩放和平移3D视图

## 使用说明

1. **启动程序**：
   ```
   python main.py
   ```

2. **加载数据**：
   - 点击"Load MRC"按钮加载原始MRC文件
   - 点击"Load Coord"按钮加载坐标文件(Coord.npy)
   - 点击"Load Input"按钮选择包含输入NPY文件的目录
   - 点击"Load Output"按钮选择包含输出NPY文件的目录

3. **初始化可视化**：
   - 点击"Initialize Visualization"按钮开始3D可视化

4. **查看和交互**：
   - 在3D视图中，绿色盒子表示有对应NPY文件的区域，红色盒子表示没有NPY文件的区域
   - 点击任意盒子可以高亮显示并查看其对应的NPY数据（如果有）
   - 在左侧列表中选择特定的NPY文件可以在3D视图中高亮对应的盒子
   - 使用"Find Box"输入框可以通过坐标查找特定的盒子

## 示例工作流程

1. 使用DiffModeler处理蛋白质结构数据，生成MRC和NPY文件
2. 启动NPY Segment Visualizer
3. 加载原始MRC文件、坐标文件和NPY目录
4. 初始化可视化，观察切割区域的分布
5. 点击感兴趣的区域，查看对应的NPY数据
6. 分析NPY数据的特征和统计信息

## 依赖库

- numpy
- PyQt5
- vtk
- mrcfile
- matplotlib

可以通过以下命令安装所需依赖：
```
pip install -r requirements.txt
```

## 技术细节

- 上位机基于PyQt5开发，提供友好的图形界面
- 使用VTK进行3D可视化，支持交互式操作
- 使用matplotlib显示NPY数据的切片和投影视图
- 数据管理基于generate_input_data.py中的切割逻辑
