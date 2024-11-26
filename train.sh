#!/bin/bash


source/usr/miniconda3/bin/activate DiffModeler

cd /share/home/xiaogenz/users/jiangzhaox/DiffModeler

#python train.py -F "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\mydataset" --info_txt "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\info_data.txt" --config config/diffmodeler_train.json --gpu 0 --output "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\train_result"
python3 /share/home/xiaogenz/users/jiangzhaox/DiffModeler/train.py -F "/share/home/xiaogenz/users/jiangzhaox/DiffModeler_data/mydataset" --info_txt "/share/home/xiaogenz/users/jiangzhaox/DiffModeler/info_data.txt" --config config/diffmodeler_train.json --gpu 0 --output "/share/home/xiaogenz/users/jiangzhaox/DiffModeler_data/train_result"