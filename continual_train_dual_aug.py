from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model, JointModel
from reid.trainer_semi_dual_aug import Trainer,eval_train
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset_semi.datasets.get_data_loaders_semi_s_w import build_data_loaders_semi, get_data_purify_shrinkmatch
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s
import os
import datetime
from matplotlib import rcParams
from scipy.interpolate import interp1d
from reid.models.promter import KernelLearning
from sklearn.cluster import DBSCAN
from reid.metric_learning.distance import cosine_similarity
from reid.utils.faiss_rerank import compute_jaccard_distance

def plot(recall, precision, i, name):
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    recall_points = np.linspace(0, 1, 100)
    f = interp1d(recall, precision, fill_value="extrapolate", kind="linear")
    precision_interp = f(recall_points)
    plt.plot(recall_points,precision_interp,color=colors[i*2], label=name)
    plt.plot(recall,precision,color=colors[i*2+1],marker="o")
    # plt.plot(recall, precision, 'ro', recall_points, precision_interp, '-b')
    # plt.plot(recall, precision, 'ro', recall_points, precision_interp, '-b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.grid(True)


def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content

def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = ("%s" % args.seed)
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    timestamp = cur_timestamp_str()
    log_name = f'log_{timestamp}.txt'
    torch.backends.cudnn.enabled = False

    # if args.resume:
    #     args.logs_dir = args.resume
    # el
    if args.evaluate:
        args.logs_dir = osp.dirname(args.evaluate)
    elif args.test_folder:
        args.logs_dir = args.test_folder
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name=f'log_res_{timestamp}.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    all_set = ['market1501', 'lpw', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'lpw', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
        training_set = ['lpw', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    elif 3 == args.setting:
        training_set = ['market1501', 'cuhk_sysu',  'msmt17', 'cuhk03']
        # all the revelent datasets
        all_set = ['market1501', 'msmt17', 'cuhk_sysu', 'cuhk03',
                'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    # all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    all_train_sets, all_test_only_sets = build_data_loaders_semi(args, training_set, testing_only_set)  

    labeled_count=0
    all_count=0

    for dd in all_train_sets:
        # train_set=dd[0].train
        for d in dd[0].train:
            if d[1]>=0:
                labeled_count+=1
            all_count+=1
    print(f"total image number {all_count}, labeled number {labeled_count}, actual label rate {labeled_count/all_count}")
    # exit(0)
    
    first_train_set = all_train_sets[0]
    model_list=[]
    for i in range(args.n_model):
        model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)
       
        model.cuda()
        model = DataParallel(model)    
        model_list.append(model)
    # writer = SummaryWriter(log_dir=args.logs_dir)
    writer = SummaryWriter(log_dir='log-output/'+osp.basename(args.logs_dir))

    start_set=0
    if args.resume_folder:
        start_set=1
        # exit(0)
        model_old_list=copy.deepcopy(model_list)
        for i in range(args.n_model):
            ckpt_name = [x + '_checkpoint-{}.pth.tar'.format(i) for x in training_set]   # obatin pretrained model name
            # print(ckpt_name[0])
            # print(args.test_folder)
            # exit(0)
            checkpoint = load_checkpoint(osp.join(args.resume_folder, ckpt_name[0]))  # load the first model
            copy_state_dict(checkpoint['state_dict'], model_list[i])     #    
            for step in range(start_set - 1):            
                model_old_list[i] = copy.deepcopy( model_list[i])    # backup the old model   
                # model_list[i].module.classifier = nn.Linear(2048, 500*(step+1 +1), bias=False) # reinitialize classifier                
                model_list[i].cuda()             
                checkpoint = load_checkpoint(osp.join(args.resume_folder, ckpt_name[step + 1]))
                copy_state_dict(checkpoint['state_dict'],  model_list[i])
                
                if args.alpha<0:                    
                    best_alpha = get_adaptive_alpha(args,  model_list[i], model_old_list[i], all_train_sets, step + 1)
                else:
                    best_alpha=args.alpha
                logger_res.append('********combining new model and old model with alpha {}********\n'.format(best_alpha))
                model_list[i] = linear_combination(args,  model_list[i], model_old_list[i], best_alpha)

                save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
                save_checkpoint({
                    'state_dict':  model_list[i].state_dict(),
                    'epoch': 0,
                    'mAP': 0,
                }, True, fpath=osp.join(args.logs_dir, save_name))

    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        model_old_list=copy.deepcopy(model_list)
        for i in range(args.n_model):
            ckpt_name = [x + '_checkpoint-{}.pth.tar'.format(i) for x in training_set]   # obatin pretrained model name
            # print(ckpt_name[0])
            # print(args.test_folder)
            # exit(0)
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
            copy_state_dict(checkpoint['state_dict'], model_list[i])     #    
            for step in range(len(ckpt_name)-1):            
                model_old_list[i] = copy.deepcopy( model_list[i])    # backup the old model   
                # model_list[i].module.classifier = nn.Linear(2048, 500*(step+1 +1), bias=False) # reinitialize classifier                
                model_list[i].cuda()             
                checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
                copy_state_dict(checkpoint['state_dict'],  model_list[i])
                
                if args.alpha<0:                    
                    best_alpha = get_adaptive_alpha(args,  model_list[i], model_old_list[i], all_train_sets, step + 1,checkpoint['psedo_dist'][i] )
                else:
                    best_alpha=args.alpha
                logger_res.append('********combining new model and old model with alpha {}********\n'.format(best_alpha))
                model_list[i] = linear_combination(args,  model_list[i], model_old_list[i], best_alpha)

                save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
                save_checkpoint({
                    'state_dict':  model_list[i].state_dict(),
                    'epoch': 0,
                    'mAP': 0,
                }, True, fpath=osp.join(args.logs_dir, save_name))
            fast_test_p_s(model_list[i], all_train_sets, all_test_only_sets, set_index=len(all_train_sets)-1, logger=logger_res,
                        args=args,writer=writer)

        exit(0)
    

    # resume from a model
    if args.resume:
        for i in range(args.n_model):           
            model_path=osp.join(args.resume, training_set[0] + '_checkpoint-{}.pth.tar'.format(i))
            checkpoint = load_checkpoint(model_path)
            copy_state_dict(checkpoint['state_dict'], model_list[i])
            start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['mAP']
            print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
            
        start_set=1
    if args.evaluate:
                
        checkpoint = load_checkpoint(args.evaluate)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
        save_name=osp.dirname(args.evaluate)+'/{}.png'.format(first_train_set[-1])
        visual_purification(model, first_train_set[4],add_num=0,save_name=save_name)
        fast_test_p_s(model_list[i], all_train_sets, all_test_only_sets, set_index=0, logger=logger_res,
                          args=args,writer=writer) 
        exit(0)
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")


    # train on the datasets squentially
    
    for set_index in range(start_set, len(training_set)):  
        if args.resume and 0==set_index:
            continue
        model_old_list=[copy.deepcopy(m) for m in model_list]     
        model_list,psedo_dist = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index,model_list, out_channel,
                                            writer,logger_res=logger_res)
        if set_index>0:
            for i in range(args.n_model):                               
                if args.alpha<0:
                    best_alpha = get_adaptive_alpha(args, model_list[i], model_old_list[i], all_train_sets, set_index,psedo_dist[i])
                else:
                    best_alpha=args.alpha
                logger_res.append('********combining new model and old model with alpha {}********\n'.format(best_alpha))
                print('********combining new model and old model with alpha {}********\n'.format(best_alpha))
                model_list[i] = linear_combination(args, model_list[i], model_old_list[i], best_alpha)           
                logger_res.append("*******testing the model-{} for {}*********".format(i+1,all_train_sets[i][-1]))
                print("*******testing the model-{} for {}*********".format(i+1,all_train_sets[i][-1])) 
                mAP =fast_test_p_s(model_list[i], all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                          args=args,writer=writer)  
    print('finished')
def get_normal_affinity(x,Norm=100):
    # from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index,psedo_dist):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]  # trainloader of current dataset
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new)
    Affin_old = get_normal_affinity(features_all_old)

    Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()

    alpha=float(1-Difference)
    return alpha

def assign_psuedo(args, model, init_loader, dataset):
    # Initialize classifer with class centers    
    class_centers = initial_classifier(model, init_loader)  # obtain the feature centers of new IDs
    Initial_Labels=initial_identity(model, init_loader, dataset,class_centers)
    # train_loader=
    Keep=torch.ones(len(Initial_Labels)).bool()
    train_loader,init_loader_new=get_data_purify_shrinkmatch(dataset, height=256, width=128, batch_size=args.batch_size,
                        workers=args.workers, num_instances=args.num_instances, Keep=Keep, Pseudo=Initial_Labels)
    return train_loader,init_loader_new
            

def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model_list, out_channel, writer,logger_res=None):
    # 获取原型
    if set_index>=1:
        _, _, _, _, prototypes, _ = extract_features_voro(model_list[-1],all_train_sets[set_index-1][-2],get_mean_feature=True)
    else:
        prototypes=None
    
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    add_num = 0
    model_old_list=[]
    if set_index>0:                
        
        for i in range(args.n_model):
            model = model_list[i]   # fetch a model
            
            class_centers = initial_classifier(model, init_loader)  # obtain the feature centers of new IDs
           
            model.module.prototype.data.copy_(class_centers)
            # model.module.classifier.weight.data[add_num:].copy_(class_centers)  # initialize the classifiers of the new IDs
            model.cuda()
            '''store the old model'''
            old_model = copy.deepcopy(model)    # copy the old model
            old_model = old_model.cuda()    # 
            old_model.eval()
            model_list[i]=model
            model_old_list.append(old_model)   
        # init_loader=init_loader_new
            

    optimizer_list=[]
    lr_scheduler_list=[]
    for i in range(args.n_model):
        model=model_list[i]
        # Re-initialize optimizer
        params = []
        for key, value in model.named_params(model):
            if not value.requires_grad:
                print('not requires_grad:', key)
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, momentum=args.momentum)    
        Stones=args.milestones
        lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

        optimizer_list.append(optimizer)
        lr_scheduler_list.append(lr_scheduler)
    
    # num_instances=len(dataset.train)
    
    if set_index>=0:
        All_Labels=[x[1] for x in dataset.train]    # 所有标签
        # print(max(All_Labels),min(All_Labels),num_classes)
        Keep=[x[1]>=0 for x in dataset.train]       # 正确标签
        train_loader,init_loader_new=get_data_purify_shrinkmatch(dataset, height=256, width=128, batch_size=args.batch_size,
                             workers=args.workers, num_instances=args.num_instances, Keep=Keep, Pseudo=All_Labels)
        
        
        trainer = Trainer(cfg, args, model_list, model_old_list, add_num + num_classes, dataset.train, All_Labels,writer=writer,prototypes=prototypes)
        
        trainer.obtain_cluster(init_loader_new, add_num,trainer.model_list, dataset_name=name) # 运行聚类算法
    if set_index>0:
        prompter=KernelLearning(n_kernel=1, groups=1, model='mobile-v3').cuda()
        checkpoint = load_checkpoint('transfer_model/{}_prompoter_49.pth.tar'.format(all_train_sets[set_index-1][-1])) 
        copy_state_dict(checkpoint['state_dict'], prompter)   
        joint_model = JointModel(args=args,model1=prompter, model2=model)
        prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(joint_model,[], init_loader, add_num=add_num)
        
        rerank_dist = compute_jaccard_distance(All_features, k1=30, k2=6)
        # select & cluster images as training set of this epochs
        cluster = DBSCAN(eps=0.5, min_samples=3, metric='precomputed', n_jobs=-1)
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        print("*********cluster number:",num_cluster)
        pseudo_labels= torch.LongTensor(pseudo_labels)
        print("ratio of out lier",(pseudo_labels==-1).float().sum()/len(Clean_FLAG)) 
        pseudo_labels[torch.where(-1==pseudo_labels)]=num_cluster            # the outliers are noted as -1 by DBSCAN
        
        pseudo_one_hot = torch.zeros(len(pseudo_labels),num_cluster+1).scatter_(1,pseudo_labels.unsqueeze(1),1)

        trainer.pseudo_labels_transfer=pseudo_labels
    else:
        trainer.psedo_iou_transfer=trainer.psedo_iou_old
        trainer.pseudo_labels_transfer=trainer.pseudo_labels_old[0]

    print('####### starting training on {} #######'.format(name))
    weight_r=[0,0]
    for epoch in range(0, Epochs):       
        train_iters=max(15,len(train_loader))
        train_loader.new_epoch()
        weight_r=trainer.train(epoch, train_loader,  optimizer_list, training_phase=set_index + 1,
                      train_iters=train_iters, add_num=add_num,weight_r=weight_r, eval_loader=init_loader_new,dataset=dataset
                      )
        
        for i in range(args.n_model):
            lr_scheduler_list[i].step()
    

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            for i in range(args.n_model):
                model=model_list[i]
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'mAP': 0.,
                }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint-{}.pth.tar'.format(name,i)))

                logger_res.append('epoch: {}'.format(epoch + 1))
                
                mAP=0.
                args.middle_test=True
                if args.middle_test and set_index==0:
                    # mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)    
                    mAP =fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                          args=args,writer=writer)                
                print("saving model to:",osp.join(args.logs_dir, '{}_checkpoint-{}.pth.tar'.format(name,i)))
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'mAP': mAP,
                    'psedo_dist':trainer.psedo_iou_old
                }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint-{}.pth.tar'.format(name,i)))    

    return model_list, trainer.psedo_iou_old

# from reid.trainer_noisy_multi_model_Purification_Rectification import eval_train
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def visual_purification(model, eval_loader,add_num,save_name, visual_num=20):
    all_loss=[]
    prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss, eval_loader, add_num)
    # prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss, eval_loader, add_num)
    recall=[]
    precise=[]
    for i in range(1,101):
        thre=i/100
        clean=torch.tensor(prob)>thre     
           
        
        recall.append(Clean_FLAG[clean].sum()/Clean_FLAG.sum())
        precise.append(Clean_FLAG[clean].sum()/(clean.sum()+1e-6))
        
        print("*********************",thre)
        print("clean ratio:{},"
            "selected data precise:{},"
            "clean data recall:{},"
            .format(Clean_FLAG.sum()/len(Clean_FLAG), 
                    Clean_FLAG[clean].sum()/(clean.sum()+1e-6),
                    Clean_FLAG[clean].sum()/Clean_FLAG.sum()
                    ))    
        print(clean.float().sum()/len(clean))            
    plot(recall, precise,0, save_name)
    return
    

def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    print('********combining new model and old model with alpha {}********\n'.format(alpha))
    if alpha <0.1:
        alpha=min(max(alpha, 0.1), 0.9)
        print('********combining new model and old model with alpha {}********\n'.format(alpha))
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,50],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', type=str, default=None, metavar='PATH',
                        help="evaluation a model")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xukunlun/DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a folder")
    parser.add_argument('--resume_folder', type=str, default=None, help="resume the models from a folder")

    parser.add_argument('--setting', type=int, default=1, choices=[1, 2,3], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")    

    parser.add_argument('--label_ratio', type=float,default=1.0, help='label_ratio')
    parser.add_argument('--alpha', type=float,default=-1, help="test during middle step")
    parser.add_argument('--spd', action='store_true', help="using the spd loss")
    parser.add_argument('--n_model', type=int,default=1, help="the number of models")
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    # parser.add_argument('--relabel', action='store_true', help="generating the pseudo label")
    parser.add_argument('--p_threshold', type=float,default=0.7, help="the threshold for pseudo label generation")
    parser.add_argument('--T_c', type=float,default=0.1, help="the threshold for old knowledge-based label filtering")
    parser.add_argument('--T_o', type=float,default=0.6, help="the threshold for new knowledge-based label filtering")
    parser.add_argument('--cluster_stride', type=int,default=5, help="the stride between cluster epochs")


    # parser.add_argument('--proto_loss', action='store_true', help="adopt the prototype for antiforgetting")
    
    main()
