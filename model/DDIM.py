from collections import OrderedDict

import torch
import torch.nn as nn
import os

from model.Base_DDIM import Base_DDIM
from model.model_utils import model_to_gpu,iou
from model.config_networks import define_G

import logging
logger = logging.getLogger('base')

class DDIM(Base_DDIM):
    def __init__(self, opt):
        super(DDIM, self).__init__(opt)
        self.netG = define_G(opt)
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
    def optimize_parameters(self, trouble_log=False, log_path=None, batch_idx=None):
        self.optG.zero_grad()
        l_pix,x_recon,x_target = self.netG(self.data)
        l_pix = l_pix.mean()
        l_pix.backward()
        self.optG.step()
        # set log
        self.log_dict['loss'] = l_pix.item()

        iou_val = iou(x_recon.sigmoid()>=0.5,x_target>0.5).mean()
        self.log_dict['iou'] = iou_val.item()
        
        # 记录问题数据的详细信息
        if trouble_log and log_path is not None and (l_pix.item() > 0.9999 or iou_val.item() < 0.0001):
            self._log_trouble_data(log_path, batch_idx, l_pix.item(), x_recon, x_target, iou_val.item())
            
        return self.log_dict

    def calculate_loss(self):
        """
        This is quick loss calculation for validation, also simply sample a timestep to get the results
        """
        self.netG.eval()
        with torch.no_grad():
            l_pix,x_recon,x_target = self.netG(self.data)
            l_pix = l_pix.mean()

        self.log_dict['loss'] = l_pix.item()
        iou_val = iou(x_recon.sigmoid()>=0.5,x_target>0.5).mean()
        self.log_dict['iou'] = iou_val.item()
        self.netG.train()
        return self.log_dict

    def test(self, continous=False):
        """
        this is really inference step by step to get the final results like we deployed for DiffModeler
        """
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                self.data['density'], continous)
            else:
                self.SR = self.netG.super_resolution(
                self.data['density'], continous)
        self.netG.train()

    def _log_trouble_data(self, log_path, batch_idx, loss_value, x_recon, x_target, iou_value):
        """记录问题数据的详细信息到日志文件"""
        with open(log_path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Problematic batch detected at batch_idx: {batch_idx}, loss: {loss_value:.6f}, iou: {iou_value:.6f}\n")
            
            # 记录输入数据信息
            if 'pid' in self.data:
                # 如果pid是列表或批次，记录整个批次的ID
                pids = self.data['pid']
                if isinstance(pids, list):
                    f.write(f"Protein IDs in batch: {', '.join(pids)}\n")
                else:
                    f.write(f"Protein ID: {pids}\n")
            
            # 记录output.npy文件路径信息
            if 'output_path' in self.data:
                output_paths = self.data['output_path']
                if isinstance(output_paths, list):
                    f.write(f"Output.npy paths:\n")
                    for i, path in enumerate(output_paths):
                        f.write(f"  [{i}] {path}\n")
                else:
                    f.write(f"Output.npy path: {output_paths}\n")
                
            # 计算并记录sigmoid后的重建数据
            x_recon_sigmoid = torch.sigmoid(x_recon)
            f.write(f"Reconstructed data stats (after sigmoid):\n")
            f.write(f"  Min value: {x_recon_sigmoid.min().item():.6f}\n")
            f.write(f"  Max value: {x_recon_sigmoid.max().item():.6f}\n")
            f.write(f"  Mean value: {x_recon_sigmoid.mean().item():.6f}\n")
            
            # 输出完整的sigmoid后数据
            f.write(f"Full sigmoid data:\n")
            # 将张量转换为numpy数组并格式化输出
            sigmoid_array = x_recon_sigmoid.cpu().detach().numpy()
            f.write(f"{sigmoid_array}\n")
