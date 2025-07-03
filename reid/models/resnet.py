import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import math
from reid.models.gem_pool import GeneralizedMeanPoolingP
from torch.nn import functional as F
import os
import cv2

def aug_map_back(x):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    x=x.detach()
    x=x.cpu()
    print(x.shape)
    x=(x)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # print(x.min(),x.max())

    x=(x*255).clamp(min=0,max=255)
    x=x.permute(0,2,3,1)
    x=x.numpy().astype('uint8')
    return x
    
def remap(inputs_r,imgs_origin, training_phase, save_dir):
    
    vis_dir=save_dir+'/vis/'+str(training_phase)+'/'
    os.makedirs(vis_dir, exist_ok=True)


    # imgs_origin=imgs_origin.permute(0,2,3,1)
    # imgs_origin=(imgs_origin*255).cpu().numpy().astype('uint8')
    # y=imgs_origin.detach()
    # y=y.cpu()
    # # print(x.shape)
    # imgs_origin=(y)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    imgs_origin=imgs_origin.permute(0,2,3,1)
    imgs_origin=(imgs_origin*255).cpu().numpy().astype('uint8')
    
    x= aug_map_back(inputs_r)
    # imgs_origin=aug_map_back(imgs_origin)

    for i in range(len(x)):
        cv2.imwrite(vis_dir+f'{i}_reconstruct.png',x[i][:,:,::-1])
        cv2.imwrite(vis_dir+f'{i}_rorigin.png',imgs_origin[i][:,:,::-1])
# 根据预测的kernel，解码预测图像
def decode_transfer_img(imgs,kernels):
    # print(imgs.size(), kernels.size())
    BS=imgs.size(0)
    for i in range(1):        
            offset=3*3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,3,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,imgs)])        
    return imgs
class JointModel(nn.Module):
    def __init__(self, args,  model1, model2, set_index=0, save_fig=False):
        super(JointModel, self).__init__()               
        self.model1 = model1           
        self.model2 = model2 
        self.args=args            
        self.mean=torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std=torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.set_index=set_index
        self.save_fig=save_fig
        
        # print(x.shape)
        # x=(x)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
    def forward(self, x, get_all_feat=False):
        origin_x =x*self.std+self.mean
        kernel=self.model1(origin_x) 
        x1=decode_transfer_img(origin_x,kernel)
        
        # 可视化重建图像
        if self.save_fig:
            remap(x1,origin_x, training_phase=self.set_index, save_dir=self.args.logs_dir)
        # exit(0)
        
        
        
        out=self.model2(x1, get_all_feat=get_all_feat)
        return out
        # if get_all_feat:
        #     return out, x1
        # else:                 
        #     return out
class Backbone(nn.Module):
    def __init__(self,last_stride, bn_norm, with_ibn, with_se,block, num_classes,layers):
        super(Backbone, self).__init__()
        self.in_planes = 2048
        self.base = ResNet(last_stride=last_stride,
                            block=block,
                            layers=layers)
        print('using resnet50 as a backbone')

        
        self.bottleneck = nn.BatchNorm2d(2048)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        # 分部分进行整流
        self.bottleneck_up=copy.deepcopy(self.bottleneck)
        self.bottleneck_down=copy.deepcopy(self.bottleneck)
        self.classifier_up=copy.deepcopy(self.classifier)
        self.classifier_down=copy.deepcopy(self.classifier)
       

        self.random_init()
        self.num_classes = num_classes
    def forward(self, x, domains=None, training_phase=None, get_all_feat=False,epoch=0):        
        x = self.base(x)
        global_feat = self.pooling_layer(x) # [16, 2048, 1, 1]
        bn_feat = self.bottleneck(global_feat) # [16, 2048, 1, 1]
        
        # global_feat=F.normalize(global_feat)    # L2 normalization
            

        if get_all_feat is True:
            cls_outputs = self.classifier(bn_feat[..., 0, 0])
            return global_feat[..., 0, 0], bn_feat[..., 0, 0], cls_outputs, x

        if self.training is False:
            # return bn_feat[..., 0, 0]
            return global_feat[..., 0, 0]

        bn_feat = bn_feat[..., 0, 0]
        cls_outputs = self.classifier(bn_feat) 

        H=x.shape[2]
        # print(x.shape)
        up_feat=self.pooling_layer(x[:,:,:H//2])
        down_feat=self.pooling_layer(x[:,:,H//2:])
        # print(down_feat.shape)

        part_out=[F.normalize(up_feat[..., 0, 0]),F.normalize(down_feat[..., 0, 0]),
                  self.classifier_up(self.bottleneck_up(up_feat)[..., 0, 0]), self.classifier_down(self.bottleneck_down(down_feat)[..., 0, 0])]


        return global_feat[..., 0, 0], bn_feat, cls_outputs, part_out

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_model(arg, num_class, camera_num, view_num,pretrain=True):
    model = Backbone(1, 'BN', False, False, Bottleneck, num_class, [3, 4, 6, 3])
    print('===========building ResNet===========')
    if pretrain:
        import torchvision
        res_base = torchvision.models.resnet50(pretrained=True)
        res_base_dict = res_base.state_dict()

        state_dict = model.base.state_dict()
        for k, v in res_base_dict.items():
            if k in state_dict:
                if v.shape == state_dict[k].shape:
                    state_dict[k] = v
                else:
                    print('param {} of shape {} does not match loaded shape {}'.format(k, v.shape,
                                                                                       state_dict[k].shape))
            else:
                print('param {} in pre-trained model does not exist in this model.base'.format(k))

        model.base.load_state_dict(state_dict, strict=True)
    return model
