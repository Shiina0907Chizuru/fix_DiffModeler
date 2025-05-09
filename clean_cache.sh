#!/bin/bash
# CentOS系统缓存清理脚本

echo "==================== 系统缓存清理 ===================="
echo "正在清理页面缓存、目录项和inode缓存..."

# 需要root权限执行以下命令
# 清理页面缓存
echo 1 > /proc/sys/vm/drop_caches

# 清理目录项和inode缓存
echo 2 > /proc/sys/vm/drop_caches

# 清理所有缓存（页面缓存、目录项和inode）
echo 3 > /proc/sys/vm/drop_caches

# 清理交换空间
swapoff -a && swapon -a

# 显示清理前后的内存使用情况
echo "==================== 清理完成 ===================="
echo "内存使用情况："
free -m

echo "磁盘使用情况："
df -h

echo "注意：此脚本需要root权限执行，命令如下："
echo "sudo bash clean_cache.sh"
