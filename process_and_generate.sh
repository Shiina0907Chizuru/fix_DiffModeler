#!/bin/bash
# 处理蛋白质数据的自动化脚本
# 作者：ZhaoXuan
# 日期：2025-03-18

# 设置工作目录 - 请根据实际情况修改
WORKDIR="/zhaoxuanj/DiffModeler"

# 设置错误处理
set -e
set -o pipefail

echo "====== 开始自动化处理蛋白质数据 $(date) ======"

# 第一步：运行processed_map.py处理地图
echo "====== 步骤1：运行processed_map.py ======"
python $WORKDIR/processed_map.py --info_txt /zhaoxuanj/20250315contour_level.txt --no_skip
echo "processed_map.py 运行完成"

# 第二步：运行generate_backbone_map.py生成骨架密度图
echo "====== 步骤2：运行generate_backbone_map.py ======"
python $WORKDIR/tools/generate_backbone_map.py --batch --data-root /zhaoxuanj/FinialPDB --protein-info /zhaoxuanj/20250315contour_levelandresolution.txt --normalize --backbone-only --force-regenerate
echo "generate_backbone_map.py 运行完成"

echo "====== 自动化处理完成 $(date) ======"
# chmod +x process_and_generate.sh
# ./process_and_generate.sh