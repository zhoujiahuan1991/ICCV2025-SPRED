from __future__ import print_function, absolute_import
import time
import os
from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *
from sklearn.mixture import GaussianMixture
from reid.utils.make_loss import make_loss, loss_fn_kd
import copy
from reid.loss.noisy_loss import LabelRefineLoss, CoRefineLoss
from sklearn.cluster import DBSCAN
from reid.metric_learning.distance import cosine_similarity
from reid.utils.faiss_rerank import compute_jaccard_distance
from scipy.interpolate import interp1d
from lreid_dataset_semi.datasets.get_data_loaders_semi_s_w import get_data_purify_shrinkmatch
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from matplotlib import rcParams
from sklearn.metrics import precision_recall_curve



class Trainer(object):
    def __init__(self,cfg,args, model_list, model_old_list, num_classes,origin_data,initial_labels,  writer=None,prototypes=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model_list = model_list
        self.model_old_list = model_old_list
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
       

        self.criterion_ce=nn.CrossEntropyLoss(reduction='none')      
       
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
        self.MSE=torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.LabelRefineLoss=LabelRefineLoss(aggregate=None)
        self.CoRefineLoss=CoRefineLoss(aggregate=None)
        self.origin_labels=initial_labels
        self.origin_labels= torch.LongTensor(self.origin_labels)
        self.refine_labels=self.origin_labels.clone()
        
        self.pseudo_labels_transfer=None  

        self.prototypes=prototypes
                

        scores_one_hot = torch.zeros(len(origin_data),num_classes).scatter_(1,self.origin_labels.unsqueeze(1).clamp(min=0),1).cuda()
        self.scores_one_hot=[scores_one_hot.clone(),scores_one_hot.clone()]
        self.pre_one_hot=scores_one_hot.clone() # predicted score

        self.cluster = DBSCAN(eps=0.5, min_samples=3, metric='precomputed', n_jobs=-1)

        self.pseudo_labels_old=[]

        self.losses=[torch.zeros(len(origin_data)).cuda(), torch.zeros(len(origin_data)).cuda()]

        self.clean_labels=[x[-1] for x in origin_data]  # clean label, not used for training, just evaluating the quality of the pseudo-labels
        self.clean_labels= torch.LongTensor(self.clean_labels).cuda()

    def eval_old_dist(self,add_num):                 
        pseudo_old = self.pseudo_labels_old[0]                  

        Ious=[]
        Labels=self.refine_labels.cpu()-add_num
        if not isinstance(pseudo_old, torch.Tensor):
            pseudo_labels=torch.tensor(pseudo_old)
        else:
            pseudo_labels=pseudo_old
        for img_id, ll in enumerate(pseudo_labels):                
            if ll<0 or Labels[img_id]<0:
                Ious.append(0)
            else:
                aa=(pseudo_labels==ll).float().sum()
                bb=(Labels==Labels[img_id]).float().sum()
                inter=(Labels[torch.where(pseudo_labels==ll)]==Labels[img_id]).float().sum()
                Ious.append((inter/(aa+bb-inter)).item())
        Ious=torch.tensor(Ious)        
        
        self.psedo_iou_old=Ious
        Thre=self.args.T_c               

        print("*********************")
        print("data keep ratio by old model:{},"                
            .format(
                    (Ious>Thre).float().sum()/Ious.size(0)
                    ))    
    def eval_transfer_dist(self,add_num):        
        pseudo_transfer = self.pseudo_labels_transfer                      

        Ious=[]
        Labels=self.refine_labels.cpu()-add_num
        if not isinstance(pseudo_transfer, torch.Tensor):
            pseudo_labels=torch.tensor(pseudo_transfer)
        else:
            pseudo_labels=pseudo_transfer
        for img_id, ll in enumerate(pseudo_labels):                
            if ll<0 or Labels[img_id]<0:
                Ious.append(0)
            else:
                aa=(pseudo_labels==ll).float().sum()
                bb=(Labels==Labels[img_id]).float().sum()
                inter=(Labels[torch.where(pseudo_labels==ll)]==Labels[img_id]).float().sum()
                Ious.append((inter/(aa+bb-inter)).item())
        Ious=torch.tensor(Ious)        
        
        self.psedo_iou_transfer=Ious
        Thre=self.args.T_o
        # print(Ious)
        print("*********************")
        print("data keep ratio by transfered old model:{},"                
            .format(
                    (Ious>Thre).float().sum()/Ious.size(0)
                    ))    
        
    def obtain_cluster(self, init_loader, add_num, model_list,dataset_name=None,res_list=None,epoch=0):
        if dataset_name:
            self.dataset_name=dataset_name
        
        self.oldmodel_filter={}
        self.pseudo_labels=[]
        self.psedo_iou=[]
        for m_id, model in enumerate(model_list):
            all_loss=[]
            if res_list is not None:
                prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=res_list[m_id]
            else:
                prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss, init_loader, add_num)
                self.Clean_FLAG=Clean_FLAG
            rerank_dist = compute_jaccard_distance(All_features, k1=30, k2=6)            
            pseudo_labels = self.cluster.fit_predict(rerank_dist)
            
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print("*********cluster number:",num_cluster)
            pseudo_labels= torch.LongTensor(pseudo_labels)
            print("ratio of out lier",(pseudo_labels==-1).float().sum()/len(Clean_FLAG)) 
            self.pseudo_labels.append(pseudo_labels)
            
            Ious=[]            
            Labels=self.refine_labels.cpu()-add_num
            if not isinstance(pseudo_labels, torch.Tensor):
                pseudo_labels=torch.tensor(pseudo_labels)            
            for img_id, ll in enumerate(pseudo_labels):                
                if ll<0 or Labels[img_id]<0:
                    Ious.append(0)
                else:
                    aa=(pseudo_labels==ll).float().sum()
                    bb=(Labels==Labels[img_id]).float().sum()
                    inter=(Labels[torch.where(pseudo_labels==ll)]==Labels[img_id]).float().sum()
                    Ious.append((inter/(aa+bb-inter)).item())
            Ious=torch.tensor(Ious)
            self.psedo_iou.append(Ious)
            

       
        if 0==epoch:
            self.pseudo_labels_old=copy.deepcopy(self.pseudo_labels)
            self.old_features=All_features
        else:
            self.eval_transfer_dist(add_num)
            
        self.eval_old_dist(add_num)    # 得到基于旧模型的预测结果与标签的差距及不确定性
        
 
        
    def decode_pre(self, model,eval_loader, add_num=0):
        
        all_loss=[]
        prob1,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss,eval_loader, add_num=add_num)  

     
        pre_ids=torch.softmax(All_logits,dim=1)[:,-500:].argmax(dim=-1)  # 预测的ID
        T_pre=pre_ids==(Clean_IDS)   # 找到预测正确的ID        
        print(
                "predicted ID precise:{},"
                "noisy ID recall:{},"
                .format(
                        T_pre.float().sum()/len(Clean_FLAG),
                        T_pre[~Clean_FLAG.bool()].float().sum()/(~Clean_FLAG.bool()).float().sum()
                        ))
        print("*********************")
        # return prob1
        return prob1,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits

    def train(self, epoch, data_loader_train,  optimizer_list, training_phase,
              train_iters=200, add_num=0, weight_r=None ,eval_loader=None ,dataset=None      
              ):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tr = [AverageMeter(),AverageMeter()]
        losses_ce_strong = [AverageMeter(),AverageMeter()]
        losses_tr_strong = [AverageMeter(),AverageMeter()]
        losses_relation = [AverageMeter(),AverageMeter()]

        if epoch>=20 :
            p_score=self.psedo_iou[0]
            Keep=p_score.cpu()>self.args.T_c      

            p_score_old=self.psedo_iou_old
            Keep_old=p_score_old.cpu()>self.args.T_o  

            p_score_transfer=self.psedo_iou_transfer
            Keep_transfer=p_score_transfer.cpu()>self.args.T_o  
            
            Keep=Keep+Keep_transfer
            # Keep=Keep_transfer


            Pseudo=(self.refine_labels-add_num) # 获得每个样本的伪标签,限制最小值
            SemiFlag=(self.origin_labels-add_num>=0).cpu()
            Keep=Keep+SemiFlag  # 正确标注一定保留
            Label_exist=self.refine_labels>=0   # 标签为-1的不要
           
            Keep=Keep*Label_exist  # 标签为-1的不要         
            
            data_loader_train,eval_loader=get_data_purify_shrinkmatch(dataset, height=256, width=128, batch_size=self.args.batch_size,
                             workers=self.args.workers, num_instances=self.args.num_instances, Keep=Keep, Pseudo=Pseudo)
            
            print("*********************")
            Is_purified_true=(self.clean_labels==Pseudo.to(self.clean_labels.device)).float()
            print(
                    "purified data precise:{},"
                    "Keeped ratio:{},"
                    "Keeped data precise:{},"
                    .format(Is_purified_true.sum()/len(Pseudo), 
                            Keep.sum()/(len(Keep)+1e-6), 
                            Is_purified_true[Keep].sum()/(Keep.sum()+1e-6),                           
                            ))
        
        if epoch%self.args.cluster_stride==0:
            res_list=[]     
            self.pre_one_hot=torch.zeros_like(self.pre_one_hot)
            
            for m_id in range(self.args.n_model):             # 先获取概率值再设置为训练模式
                res=self.decode_pre(self.model_list[m_id],eval_loader, add_num)
                res_list.append(res)   # 获得概率值
                self.pre_one_hot+=torch.softmax(res[-1].cuda(), -1)
            self.pre_one_hot=self.pre_one_hot/self.args.n_model

 

            for idx, pred in enumerate(self.pre_one_hot):
                if self.origin_labels[idx]>=add_num:    
                    pass # remain the labels of labeled data
                else:
                    values, indices = torch.topk(pred, 2)
                    values=values/values.sum()
                    
                    if values.max()>self.args.p_threshold and torch.argmax(pred)>=add_num:
                        self.refine_labels[idx]=torch.argmax(pred).cpu()
                    else:
                        self.refine_labels[idx]=-1



            if training_phase>0 and epoch>0 and (self.args.T_c<1.0 or epoch<10):
                self.obtain_cluster(eval_loader, add_num, self.model_list, res_list=res_list,epoch=epoch)
                self.psedo_iou[0]=self.psedo_iou[0].cuda()
                

   

        self.model_list[0].train()       
        
        for m_id in range(self.args.n_model):
            # freeze the bn layer totally
            for m in self.model_list[m_id].module.base.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.requires_grad == False and m.bias.requires_grad == False:
                        m.eval()    
            
            
        end = time.time()  
        train_iters=max(15,len(data_loader_train))
        for i in range(train_iters):    
            try:            
                train_inputs = data_loader_train.next()
            except:
                continue
            data_time.update(time.time() - end)
        
            s_inputs,s_inputs_strong, targets, cids, image_id,clean_pid=self._parse_data(train_inputs)
        
            indexes=image_id

            targets += add_num
            s_inputs_all=torch.cat((s_inputs,s_inputs_strong))
            BS=s_inputs.size(0)
            s_features_1, bn_feat_1, cls_outputs_1, feat_final_layer_1 = self.model_list[0](s_inputs_all)      
            loss_ce1, loss_tp_1 = self.loss_fn(cls_outputs_1[:BS], s_features_1[:BS], targets, target_cam=None)
                 
            loss_ce1_strong, loss_tp_1_strong = self.loss_fn(cls_outputs_1[BS:], s_features_1[BS:], targets, target_cam=None)
            cls_outputs_1, s_features_1=cls_outputs_1[:BS], s_features_1[:BS]
            
            loss_1 = loss_ce1+loss_tp_1 +loss_ce1_strong+loss_tp_1_strong
           
            
            losses_ce[0].update(loss_ce1.item(), s_inputs.size(0))
            losses_ce_strong[0].update(loss_ce1_strong.item(), s_inputs.size(0))
            losses_tr[0].update(loss_tp_1.item(), s_inputs.size(0))
            losses_tr_strong[0].update(loss_tp_1_strong.item(), s_inputs.size(0))
               

            if len(self.model_old_list):                                
                Keep1=None
                                           
                af_loss_1=self.anti_forgetting(self.model_old_list[0],s_inputs,cls_outputs_1,s_features_1, targets,feat_final_layer_1,Keep1)
                

                loss_1+=af_loss_1

                losses_relation[0].update(af_loss_1.item(), s_inputs.size(0))

                                                
         

            optimizer_list[0].zero_grad()
            loss_1.backward()
            optimizer_list[0].step()      

   

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce[0].val,
                        global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr[0].val,
                        global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                        global_step=epoch * train_iters + i)
           
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Loss_ce1 {:.3f} ({:.3f})\t'
                    'Loss_tp1 {:.3f} ({:.3f}) \t'
                    'Loss_ce1_strong {:.3f} ({:.3f})\t'
                    'Loss_tp1_strong {:.3f} ({:.3f}) \t'
                    'Loss_relation1 {:.3f} ({:.3f}) \t'
                    .format(epoch, i + 1, train_iters,
                            batch_time.val, batch_time.avg,
                            losses_ce[0].val, losses_ce[0].avg,
                            losses_tr[0].val, losses_tr[0].avg,
                            losses_ce_strong[0].val, losses_ce_strong[0].avg,
                            losses_tr_strong[0].val, losses_tr_strong[0].avg,
                            losses_relation[0].val, losses_relation[0].avg,
                ))     

        weight_r = [1. / (1. + losses_ce[0].avg)]   
        return  weight_r

    def get_normal_affinity(self,x,y, Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,y)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs,imgs_strong, image_id, pids, cids, clean_pid = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        imgs_strong=imgs_strong.cuda()
        
        return inputs, imgs_strong, targets, cids, image_id, clean_pid
    
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old,targets):        
        Target=Affinity_matrix_old
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)

        return divergence.sum(-1)

    
    def anti_forgetting(self, old_model,s_inputs,cls_outputs,s_features, targets,feat_final_layer, Keep=None):
        divergence=0
        loss=0
        old_model.eval()
        with torch.no_grad():            
            s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
        if isinstance(s_features_old, tuple):
            s_features_old=s_features_old[0]
       
      
        Affinity_matrix_new = self.get_normal_affinity(s_features,self.prototypes)
        Affinity_matrix_old = self.get_normal_affinity(s_features_old,self.prototypes)
        divergence = self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)

        divergence=divergence.mean()
                              
        loss = loss + divergence * self.AF_weight
        return loss


def eval_train(model,all_loss, eval_loader, add_num=0):  
    CE = nn.CrossEntropyLoss(reduction='none').cuda()  
    model.eval()
    losses = torch.zeros(50000)    
    Clean_IDS=torch.zeros(50000)
    Noisy_IDS=torch.zeros(50000)
    Clean_FLAG=torch.zeros(50000)
    All_features=torch.zeros(50000,2048)
    All_logits=torch.zeros(50000,500+add_num)
    Count=0
    "img, image_id, pid, camid, clean_pid"
    with torch.no_grad():
        for i, (imgs, image_id, pids, cids, clean_pid) in enumerate(eval_loader):
        # for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            index=image_id
            inputs=imgs
            targets=pids+add_num
            inputs, targets = inputs.cuda(), targets.cuda() 
            # _, _, outputs = model(inputs) 
            s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = model(inputs, get_all_feat=True)
            Count+=len(imgs)
            loss = CE(cls_outputs_old, targets.clamp(min=0))  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]    
                Clean_IDS[index[b]]=clean_pid[b]     
                Noisy_IDS[index[b]]=pids[b] 
                Clean_FLAG[index[b]]=clean_pid[b] ==  pids[b] 
                All_features[index[b]]=s_features_old[b].detach().cpu().clone()
                All_logits[index[b]]=cls_outputs_old[b].detach().cpu().clone()
    losses=losses[:Count]
    Clean_IDS=Clean_IDS[:Count]
    Noisy_IDS=Noisy_IDS[:Count]
    Clean_FLAG=Clean_FLAG[:Count]
    All_features=All_features[:Count]
    All_logits=All_logits[:Count]
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9:
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    #     input_loss = losses.reshape(-1,1)
    input_loss = losses.reshape(-1,1)
    
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits

