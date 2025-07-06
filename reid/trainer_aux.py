from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy
from reid.utils.color_transformer import ColorTransformer
from reid.metric_learning.distance import cosine_similarity
import random
import cv2
import os

def remap(inputs_r,imgs_origin, training_phase, save_dir,dataset_name):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    x=inputs_r.detach()
    x=x.cpu()
    # print(x.shape)
    x=(x)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # print(x.min(),x.max())

    x=(x*255).clamp(min=0,max=255)
    x=x.permute(0,2,3,1)
    x=x.numpy().astype('uint8')
    vis_dir=save_dir+f'/vis/{dataset_name}/'+str(training_phase)+'/'
    os.makedirs(vis_dir, exist_ok=True)


    imgs_origin=imgs_origin.permute(0,2,3,1)
    imgs_origin=(imgs_origin*255).cpu().numpy().astype('uint8')
    

    for i in range(len(x)):
        cv2.imwrite(vis_dir+f'{i}_reconstruct.png',x[i][:,:,::-1])
        cv2.imwrite(vis_dir+f'{i}_rorigin.png',imgs_origin[i][:,:,::-1])
    print("saved images in ", vis_dir)
    

class RehearserTrainer(object):
    def __init__(self,args=None, rehearser=None):
        super(RehearserTrainer, self).__init__()
        self.color_transformer = ColorTransformer()
        self.train_transformer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.rehearser = rehearser
        self.args =args

        self.MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    
    def train(self, epoch, loader_source, train_iters=200,dataset_name=None,
              print_freq=100, writer=None):
        losses_cond = AverageMeter()

        for it in range(train_iters):
            _,s_inputs_o, _ = self._parse_data(
            loader_source.next()
            )

            trans_inputs = self.color_transformer.color_transfer_resample(s_inputs_o, lab=False)    # distribution augmentation
            if self.args.blur and random.random()>0.5: # random blur
                trans_inputs=self.blur(trans_inputs)
            kernel = self.rehearser(trans_inputs)    # learn the convolution kernel
            inputs_r=decode_transfer_img(self.args,trans_inputs,kernel)
            # # conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            # k_w=kernel[:,:3*3*3*3]
            # BS=kernel.size(0)
            # k_w=k_w.reshape(BS,3,3,3,3)
            # k_b=kernel[:,3*3*3*3:]
            # inputs_r=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,trans_inputs)])
            target_inputs=self.train_transformer(s_inputs_o)    
            loss_c=self.MAE(inputs_r, target_inputs)
            
            losses_cond.update(loss_c.item())

            self.rehearser.optim.zero_grad()
            loss_c.backward()
            self.rehearser.optim.step()

            if (it + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss_cond {:.3f} ({:.3f})'.format(
                epoch, it + 1, train_iters,
                losses_cond.val, losses_cond.avg))
                if it<print_freq:
                    remap(inputs_r,s_inputs_o, epoch, self.args.logs_dir, dataset_name)

    def _parse_data(self, inputs):
        imgs,imgs_o, _, pids, _, _ = inputs
        # print(inputs)
        imgs_o = imgs_o.cuda()
        imgs = imgs.cuda()
        pids = pids.cuda() 
        return imgs,imgs_o, pids
# decode the reconstructed images according to the predicted instance-specific kernels
def decode_transfer_img(args,imgs,kernels):
    BS=imgs.size(0)
    for i in range(args.n_kernel):
        if args.groups==1:
            offset=3*3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,3,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,imgs)])
        elif args.groups==3:
            offset=3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,1,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1,groups=args.groups) for w,b,img in zip(k_w,k_b,imgs)])
        else:
            raise Exception(f"The learned convolution group number \'groups={args.groups}\' is not supported!") 

    return imgs

        
        
        
        
        
                