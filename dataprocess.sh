#!/bin/bash

source/usr/miniconda3/bin/activate DiffModeler

cd /share/home/xiaogenz/users/jiangzhaox/DiffModeler

python3 /share/home/xiaogenz/users/jiangzhaox/DiffModeler/preprocess_protein.py --info_txt "/share/home/xiaogenz/users/jiangzhaox/DiffModeler/info.txt"
python3 /share/home/xiaogenz/users/jiangzhaox/DiffModeler/generate_dataset.py --info_txt "/share/home/xiaogenz/users/jiangzhaox/DiffModeler/info.txt"
python3 /share/home/xiaogenz/users/jiangzhaox/DiffModeler/make_OnlyCAatomslabel.py --info_txt "/share/home/xiaogenz/users/jiangzhaox/DiffModeler/info.txt"
