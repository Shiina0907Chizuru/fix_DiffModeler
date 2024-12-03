from collections import OrderedDict
import torch
import torch.nn as nn
import os
from model.BaseNetwork import BaseNetwork
from model.model_utils import model_to_gpu, iou
from model.config_networks_new import define_network

import logging
logger = logging.getLogger('base')

class NetworkTemplate(BaseNetwork):
    def __init__(self, opt):
        super(NetworkTemplate, self).__init__(opt)
        self.netG = define_network(opt)
        self.netG = model_to_gpu(self.netG)
        self.schedule_phase = None
        # set loss and load resume state
        self.set_loss()
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
        else:
            self.netG.eval()
        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss()
        else:
            self.netG.set_loss()

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix, x_recon, x_target = self.netG(self.data)
        l_pix = l_pix.mean()
        l_pix.backward()
        self.optG.step()
        # set log
        self.log_dict['loss'] = l_pix.item()
        iou_val = iou(x_recon.sigmoid()>=0.5, x_target>0.5).mean()
        self.log_dict['iou'] = iou_val.item()
        return self.log_dict

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['density'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['density'], continous)
        self.netG.train()

    def calculate_loss(self):
        self.netG.eval()
        with torch.no_grad():
            l_pix, x_recon, x_target = self.netG(self.data)
            l_pix = l_pix.mean()
        self.log_dict['loss'] = l_pix.item()
        iou_val = iou(x_recon.sigmoid()>=0.5, x_target>0.5).mean()
        self.log_dict['iou'] = iou_val.item()
        self.netG.train()
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                out_dict[k] = v.detach().cpu()
        return out_dict

    def get_current_losses(self):
        return self.log_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                           self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
