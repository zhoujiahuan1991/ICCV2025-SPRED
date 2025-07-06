# import torchvision.transforms as T_T
import copy
import os.path
import os
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.feature_tools import *
import lreid_dataset_semi.datasets as datasets
# from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.sampler import RandomMultipleGallerySampler,RandomIdentitySampler
from reid.utils.data import IterLoader
import collections
import numpy as np
import random
import copy
import os.path as osp
from PIL import ImageOps, ImageFilter
from torchvision.transforms import RandAugment
from torchvision.transforms import transforms
from torchvision.transforms import RandomErasing
import PIL.ImageDraw
from .ShrinkMatchTransfer import RandAugmentMC


name_map={
    'market1501':"market", 
    'cuhk_sysu':"subcuhksysu", 
    'dukemtmc':"duke", 
    'msmt17':"msmt17", 
    'cuhk03':"cuhk03",
    'lpw':"lpw"
}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_data_semi(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0,label_ratio=1.0, min_instance=2, training=False):
    root = data_dir
    dataset = datasets.create(name, root)
    # create the clean data first
    if training:
        if select_num > 0:
            '''select some persons for training'''
            train = []
            for instance in dataset.train:
                if instance[1] < select_num:
                    train.append((instance[0], instance[1], instance[2], instance[3]))  #img_path, pid, camid, domain-id

            dataset.train = train
            dataset.num_train_pids = select_num
            dataset.num_train_imgs = len(train)

            '''using semi labels'''            
            file_path='semi_labeled_data/{}_{}.pt'.format(name_map[name],label_ratio)
        else:
            file_path='semi_labeled_data_full/{}_{}.pt'.format(name_map[name],label_ratio)

        if osp.isfile(file_path):
            print("**************\nloading semi data form {}\n********* ".format(file_path))
            semi_data=torch.load(file_path)
            train = []        
                
            labeled_count=0
            for idx, d in enumerate(semi_data):   #img_path, pid, camid, domain-id, image-id, clean_pid
                # clean_pid=dataset.train[idx][1]
                clean_pid=d[4]

                img_path=osp.join(osp.dirname(dataset.train[idx][0]), d[0])

                # print(d, dataset.train[idx])
                
                train.append((img_path,d[1],d[2],dataset.train[0][3],d[4], d[5]))#img_path, pid, camid, domain-id, image-id, clean_pid
                if d[-1]==clean_pid:
                    labeled_count+=1                
            print("labeled ratio:",labeled_count/len(train))        
            dataset.train = train
            
        else:
            labeled_count=0
            # 打乱顺序后每个ID的第一幅图一定保留，设定保留第一个后，每幅图标注被保留的概率
            train_origin=copy.deepcopy(dataset.train)
            random.shuffle(train_origin)
            label_ratio=(label_ratio*len(train_origin)-dataset.num_train_pids*min_instance)/(len(train_origin)-min_instance*dataset.num_train_pids)
            # label_ratio=label_ratio-dataset.num_train_pids/len(train_origin) 
            # recorded_id=[]  # 已经见过的ID
            recorded_id = collections.defaultdict(list)
            train_new=[]
            for idx, d in enumerate(train_origin):   #img_path, pid, camid, domain-id
                pid=d[1]
                if (len(recorded_id[pid])>=min_instance ) and (random.random()>label_ratio):
                    train_new.append((d[0],-1,d[2],d[3],idx,d[1]))
                else:
                    train_new.append((d[0],d[1],d[2],d[3],idx,d[1]))#img_path, pid, camid, domain-id, image-id, clean_pid
                    labeled_count+=1
                recorded_id[pid].append(train_new[-1])
                # if (pid in recorded_id) and (random.random()>label_ratio):   # 除了第一个ID，其余按比例抽样             
                #     train_new.append((d[0],-1,d[2],d[3],idx,d[1]))
                # else:
                #     recorded_id.append(pid)
                #     train_new.append((d[0],d[1],d[2],d[3],idx,d[1]))#img_path, pid, camid, domain-id, image-id, clean_pid
                #     labeled_count+=1
            os.makedirs(osp.dirname(file_path), exist_ok=True)
            torch.save(train_new, file_path)
            print("labeled ratio:",labeled_count/len(train))        
            dataset.train = train_new

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
       

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    basic_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer,transform_strong=basic_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]





def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def build_data_loaders(cfg, training_set, testing_only_set, select_num=500):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data_semi(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, select_num=select_num, label_ratio=1.0, training=True) for name in training_set]

  
    testing_loaders = [get_data_semi(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders
