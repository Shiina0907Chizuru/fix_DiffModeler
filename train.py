

import os
from ops.argparser import argparser_train

if __name__ == "__main__":
    params = argparser_train()
    gpu_id = params['gpu']
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from training.main_worker import main_worker
    main_worker(params)
# command: python train.py -F "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\mydataset" --info_txt "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler\info_data.txt" --config config/diffmodeler_train.json --gpu 0 --output "E:\ZJUT\Research\MrZhouDeepLearning\DiffReaserch\DiffModeler_data\train_result"