import os
import json
import torch
import logging
from model.config_networks_new import define_network
from ops.os_operation import mkdir
from data_processing.Test_Dataset import Single_Dataset

logger = logging.getLogger('base')

def setup_logger(log_path):
    """设置日志"""
    log_format = '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def train(config_path):
    """训练网络
    
    Args:
        config_path (str): 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建保存目录
    save_dir = os.path.join('experiments', config['name'])
    mkdir(save_dir)
    
    # 设置日志
    setup_logger(os.path.join(save_dir, 'train.log'))
    logger.info('Config loaded.')
    
    # 创建网络
    network = define_network(config)
    logger.info('Network created.')
    
    # 设置数据集
    train_dataset = Single_Dataset(config['data']['train_dir'], "train_")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['model']['num_workers'],
        pin_memory=True
    )
    
    # 训练循环
    total_epochs = config['train']['total_epochs']
    save_freq = config['train']['save_freq']
    
    for epoch in range(total_epochs):
        # 训练一个epoch
        for idx, data in enumerate(train_loader):
            network.feed_data(data)
            log_dict = network.optimize_parameters()
            
            # 打印训练信息
            if idx % 10 == 0:
                msg = f'Epoch: [{epoch}/{total_epochs}], Iter: [{idx}/{len(train_loader)}]'
                for k, v in log_dict.items():
                    msg += f', {k}: {v:.4f}'
                logger.info(msg)
        
        # 更新学习率
        network.update_learning_rate()
        
        # 保存模型
        if (epoch + 1) % save_freq == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            network.save_network(save_path, epoch, idx)
            logger.info(f'Model saved to {save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
