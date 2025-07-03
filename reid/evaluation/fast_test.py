import torch
import time
from reid.evaluation import (fast_evaluate_rank, compute_distance_matrix)
class CatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()
def time_now():
    '''return current time in format of 2000-01-01 12:01:01'''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def fast_test_p_s(model, all_train_sets, all_test_only_sets,set_index, args, logger=None,writer=None):
    # using Cython test during train
    # return mAP, Rank-1
    print(f'****** start perform fast testing! ******')
    loaders=all_train_sets[:set_index+1]+all_test_only_sets
    # meters
    # compute query and gallery features
    def _cmc_map(_query_features_meter, _gallery_features_meter):
        query_features = _query_features_meter.get_val()
        gallery_features = _gallery_features_meter.get_val()

        distance_matrix = compute_distance_matrix(query_features, gallery_features, 'euclidean')
        distance_matrix = distance_matrix.data.cpu().numpy()
        use_cython=False if args.save_evaluation else True
        try:
            save_dir=args.logs_dir
        except:
            save_dir=args.log_dir
        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=use_cython,save_dir=save_dir )

        return CMC[0] * 100, mAP * 100
    results_dict = {}
    model.eval()

    for temp_loaders in loaders:
        dataset, num_classes, train_loader, test_loader, init_loader, dataset_name=temp_loaders
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

        features_meter, pids_meter, cids_meter = CatMeter(), CatMeter(), CatMeter()
        file_names_all={}
        f_count=0

        torch.cuda.empty_cache()
        print(time_now(), f' {dataset_name} feature start ')
        with torch.no_grad():
            for i, (imgs, fnames, pids, cids, domains) in enumerate(test_loader):
                images, pids, cids=imgs,pids,cids
                images = images.cuda()
                features = model(images)

                features_meter.update(features.data)
                pids_meter.update(pids)
                cids_meter.update(cids)
                for fname in fnames:
                    file_names_all[fname]=f_count
                    f_count+=1


        x = [file_names_all[f] for f, _, _, _ in dataset.query]
        y = [file_names_all[f] for f, _, _, _ in dataset.gallery]
        query_features_meter.val, query_pids_meter.val, query_cids_meter.val=features_meter.val[x], pids_meter.val[x], cids_meter.val[x]
        gallery_features_meter.val, gallery_pids_meter.val, gallery_cids_meter.val = features_meter.val[y], pids_meter.val[y], cids_meter.val[y]

        print(time_now(), f' {dataset_name} feature done')
        torch.cuda.empty_cache()

        # proto_type = {
        #     'query_features': query_features_meter.get_val(),
        #     'query_labels': query_pids_meter.get_val(),
        #     'gallery_features': gallery_features_meter.get_val(),
        #     'gallery_labels': gallery_pids_meter.get_val(),
        # }
        # save_path = config.resume_test_model + '/{}.pt'.format(dataset_name)
        # print("saving features of dataset {} to {}".format(dataset_name, save_path))
        # torch.save(proto_type, save_path)

        # print(time_now(), f' {dataset_name} feature done')

        rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        print("mAP/Rank1:\t{:.1f}/{:.1f}".format(map, rank1))
        # rank1, map = _cmc_map(query_features_meter, gallery_features_meter)
        results_dict[f'{dataset_name}_mAP'], results_dict[f'{dataset_name}_Rank1'] = map, rank1

        if writer is not None:
            writer.add_scalar(tag="results/{}_mAP".format(dataset_name), scalar_value=map,
                               global_step=set_index)
            writer.add_scalar(tag="results/{}_R@1".format(dataset_name), scalar_value=rank1,
                              global_step=set_index)

    results_str = ''
    for criterion, value in results_dict.items():
        results_str = results_str + f'\n{criterion}: {value}'
    aver_seen_mAP, aver_seen_r1, aver_unseen_mAP, aver_unseen_r1=print_results(results_dict, all_train_sets, all_test_only_sets, set_index, logger=logger)


    if writer is not None:
            writer.add_scalar(tag="results/Seen-Avg_mAP", scalar_value=aver_seen_mAP,
                               global_step=set_index)
            writer.add_scalar(tag="results/Seen-Avg_R@1", scalar_value=aver_seen_r1,
                              global_step=set_index)
            
            writer.add_scalar(tag="results/UnSeen-Avg_mAP", scalar_value=aver_unseen_mAP,
                               global_step=set_index)
            writer.add_scalar(tag="results/UnSeen-Avg_R@1", scalar_value=aver_unseen_r1,
                              global_step=set_index)
    return map

def print_results(rank_map_dict, all_train_sets, all_test_only_sets,set_index, logger=None):
    seen_r1 = []
    seen_map = []
    unseen_r1 = []
    unseen_map = []
    names = ''
    Results = ''
    for temp_loaders in all_train_sets[:set_index+1]:
        dataset, num_classes, train_loader, test_loader, init_loader, name = temp_loaders
        seen_r1.append(rank_map_dict[f'{name}_Rank1'])
        seen_map.append(rank_map_dict[f'{name}_mAP'])

        names = names + name + '\t\t'
        Results = Results + '|{:.1f}/{:.1f}\t'.format(float(seen_map[-1]),float(seen_r1[-1]) )
    names_unseen = ''
    Results_unseen = ''
    for temp_loaders in all_test_only_sets:
        dataset, num_classes, train_loader, test_loader, init_loader, name = temp_loaders
        unseen_r1.append(rank_map_dict[f'{name}_Rank1'])
        unseen_map.append(rank_map_dict[f'{name}_mAP'])

        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(float(unseen_map[-1]),float(unseen_r1[-1]), )
    # for name in config.test_dataset:
    #     if name in config.train_dataset:
    #         seen_r1.append(rank_map_dict[f'{name}_Rank1'])
    #         seen_map.append(rank_map_dict[f'{name}_mAP'])
    #     else:
    #         unseen_r1.append(rank_map_dict[f'{name}_Rank1'])
    #         unseen_map.append(rank_map_dict[f'{name}_mAP'])
    import numpy as np
    aver_seen_mAP=np.round(np.mean(seen_map), 1)
    aver_seen_r1=np.round(np.mean(seen_r1),1)
    aver_unseen_mAP = np.round(np.mean(unseen_map), 1)
    aver_unseen_r1 = np.round(np.mean(unseen_r1), 1)
    print("Average mAP on Seen dataset:", aver_seen_mAP)
    print("Average R1 on Seen dataset:", aver_seen_r1)
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_seen_mAP, aver_seen_r1)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on UnSeen dataset:", aver_unseen_mAP)
    print("Average R1 on UnSeen dataset:", aver_unseen_r1)
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_unseen_mAP , aver_unseen_r1 )
    print(names_unseen)
    print(Results_unseen)
    print(f"{aver_seen_mAP}\t{aver_seen_r1}\t{aver_unseen_mAP}\t{aver_unseen_r1}")


    # print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    # print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    

    if logger:
        # logger.info()
        try:
            logger.info(names)
            logger.info(Results)
            logger.info(Results.replace('|', '').replace('/', '\t'))
            logger.info(names_unseen)
            logger.info(Results_unseen)
            logger.info(Results_unseen.replace('|', '').replace('/', '\t'))
        except:
            logger.append(names)
            logger.append(Results)
            logger.append(Results.replace('|', '').replace('/', '\t'))
            logger.append(names_unseen)
            logger.append(Results_unseen)
            logger.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return aver_seen_mAP, aver_seen_r1, aver_unseen_mAP, aver_unseen_r1

    # logger("{} {} ".format("seen_map:", np.round(np.mean(seen_map), 1)))
    # logger("{} {} ".format("seen_r1:", np.round(np.mean(seen_r1))))
    # logger("{} {} ".format("unseen_map:", np.round(np.mean(unseen_map))))
    # logger("{} {} ".format("unseen_r1:", np.round(np.mean(unseen_r1))))
    #
    # logger(
    #     f"{np.mean(seen_map).round(1)}\t{np.mean(seen_r1).round(1)}\t{np.mean(unseen_map).round(1)}\t{np.mean(unseen_r1).round(1)}")