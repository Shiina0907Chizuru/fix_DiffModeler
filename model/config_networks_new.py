import logging
from model.NetworkTemplate import NetworkTemplate
logger = logging.getLogger('base')

def define_network(opt):
    """Define and create the network.
    
    Args:
        opt (dict): Network options, should contain:
            model:
                type: network type
                network: network architecture parameters
            phase: 'train' or 'test'
            train: (optional) training parameters
    
    Returns:
        BaseNetwork: Created network
    """
    # 根据配置选择网络类型
    net_type = opt['model']['type']
    
    if net_type == 'template':
        net = NetworkTemplate(opt)
    # 在这里添加新的网络类型
    # elif net_type == 'your_network':
    #     from model.your_network import YourNetwork
    #     net = YourNetwork(opt)
    else:
        raise NotImplementedError(f'Network type {net_type} not implemented')
    
    logger.info('Network created.')
    return net
