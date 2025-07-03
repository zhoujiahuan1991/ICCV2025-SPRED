CUDA_VISIBLE_DEVICES=6 python continual_train_dual_aug.py --logs-dir logs_dual-aug/label-0.1 --label_ratio 0.1   

CUDA_VISIBLE_DEVICES=7 python continual_train_dual_aug.py --logs-dir logs_dual-aug/label-0.2 --label_ratio 0.2  

CUDA_VISIBLE_DEVICES=5 python continual_train_dual_aug.py --logs-dir logs_dual-aug/label-0.5 --label_ratio 0.5  
