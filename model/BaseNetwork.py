import os
import torch
import torch.nn as nn

class BaseNetwork():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        """处理输入数据
        Args:
            data (dict): 包含输入数据的字典
        """
        pass

    def optimize_parameters(self):
        """执行一步训练"""
        pass

    def get_current_visuals(self):
        """获取当前可视化结果"""
        pass

    def get_current_losses(self):
        """获取当前损失值"""
        pass

    def print_network(self):
        """打印网络结构"""
        pass

    def get_network_description(self, network):
        '''获取网络的字符串描述和参数总数'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def set_device(self, x):
        """将数据转移到指定设备"""
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x
