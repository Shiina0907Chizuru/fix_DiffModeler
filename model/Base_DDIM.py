import os
import torch
import torch.nn as nn


class Base_DDIM():
    def __init__(self, opt):
        self.opt = opt
        self.begin_step = 0
        self.begin_epoch = 0
        self.device = torch.device('cuda')

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    # 处理字典值是列表的情况
                    if isinstance(item, list):
                        device_list = []
                        for subitem in item:
                            if subitem is not None:
                                # 检查subitem是否有to方法(张量有，字符串等没有)
                                if hasattr(subitem, 'to'):
                                    device_list.append(subitem.to(self.device))
                                else:
                                    # 如果没有to方法，保持原样
                                    device_list.append(subitem)
                            else:
                                device_list.append(None)
                        x[key] = device_list
                    else:
                        # 检查item是否有to方法
                        if hasattr(item, 'to'):
                            x[key] = item.to(self.device)
        elif isinstance(x, list):
            # 创建一个新的列表来保存移动到设备的元素
            device_list = []
            for item in x:
                if item is not None:
                    # 检查item是否有to方法
                    if hasattr(item, 'to'):
                        device_list.append(item.to(self.device))
                    else:
                        # 如果没有to方法，保持原样
                        device_list.append(item)
                else:
                    device_list.append(None)
            x = device_list  # 将原列表替换为新列表
        else:
            # 检查x是否有to方法
            if hasattr(x, 'to'):
                x = x.to(self.device)
        return x