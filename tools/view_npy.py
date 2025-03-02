import numpy as np
import sys
import os

def view_npy(file_path):
    """查看npy文件的内容
    Args:
        file_path: npy文件路径
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
        
    try:
        data = np.load(file_path)
        print("\n文件信息:")
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"最小值: {data.min()}")
        print(f"最大值: {data.max()}")
        print(f"平均值: {data.mean()}")
        print(f"标准差: {data.std()}")
        print(f"非零元素数量: {np.count_nonzero(data)}")
        print(f"总元素数量: {data.size}")
        
        print("\n完整数据内容:")
        # 设置numpy打印选项，显示所有元素
        np.set_printoptions(threshold=np.inf, precision=4, suppress=True)
        print(data)
        
        # 重置打印选项
        np.set_printoptions(threshold=1000)
        
    except Exception as e:
        print(f"错误: 无法加载文件 - {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python view_npy.py <npy文件路径>")
    else:
        view_npy(sys.argv[1])
