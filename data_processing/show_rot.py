import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义basic_rot_ax函数
def basic_rot_ax(m, ax=0):
    ax %= 3  # 限制轴在 0, 1, 2 范围内
    if ax == 0:
        return np.rot90(m[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2), 3)
    if ax == 1:
        return np.rot90(m, 1)
    if ax == 2:
        return m.swapaxes(0, 2)[::-1, :, :]


# 绘制3D数组
def plot_3d_array_with_planes(data, title, ax):
    ax.set_title(title)
    ax.set_xlabel('First Dimension')  # 修改标签
    ax.set_ylabel('Second Dimension')
    ax.set_zlabel('Third Dimension')

    # 设置立方体比例
    ax.set_box_aspect([1, 1, 1])  # 确保各轴比例一致

    # 使用不同的颜色标注每个 xz 平面
    colors = ['red', 'blue', 'green']
    for i in range(data.shape[0]):  # 遍历第一个维度
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                ax.scatter(i, j, k, c=colors[i], s=100, alpha=0.7)  # 给每个 xz 平面设置颜色
                ax.text(i, j, k, f'{data[i, j, k]}', color='black')  # 数值标注

# def print_3d_array(array, label):
#     """打印3D数组的内容，并标注描述"""
#     print(f"{label}:")
#     for i, matrix in enumerate(array):
#         print(f"Layer {i}:\n{matrix}")
#     print("\n")
def print_3d_array(array, label):
    """以格式化方式打印3D数组"""
    print(f"{label}:\n")
    for i, matrix in enumerate(array):
        print(matrix)
        if i < len(array) - 1:  # 每个层之间加一个空行
            print()
    print("-" * 30)  # 分隔线

# 原始数组
m = np.arange(27).reshape(3, 3, 3)

# 沿不同轴旋转
rot_ax_0 = basic_rot_ax(m, ax=0)
rot_ax_1 = basic_rot_ax(m, ax=1)
rot_ax_2 = basic_rot_ax(m, ax=2)

print_3d_array(m, "Original Array")
print_3d_array(rot_ax_0, "Rotated along Axis 0")
print_3d_array(rot_ax_1, "Rotated along Axis 1")
print_3d_array(rot_ax_2, "Rotated along Axis 2")

# 可视化
fig = plt.figure(figsize=(18, 6))

# 原始数组
ax1 = fig.add_subplot(141, projection='3d')
plot_3d_array_with_planes(m, "Original Array", ax1)

# 沿第0轴旋转
ax2 = fig.add_subplot(142, projection='3d')
plot_3d_array_with_planes(rot_ax_0, "Rotated along Axis 0", ax2)

# 沿第1轴旋转
ax3 = fig.add_subplot(143, projection='3d')
plot_3d_array_with_planes(rot_ax_1, "Rotated along Axis 1", ax3)

# 沿第2轴旋转
ax4 = fig.add_subplot(144, projection='3d')
plot_3d_array_with_planes(rot_ax_2, "Rotated along Axis 2", ax4)

plt.tight_layout()
plt.show()