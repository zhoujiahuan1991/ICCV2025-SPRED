
CUDA_VISIBLE_DEVICES=1 python continual_train.py --logs-dir logs/label-0.1 --label_ratio 0.1  
CUDA_VISIBLE_DEVICES=1 python continual_train.py --logs-dir logs/label-0.2 --label_ratio 0.2  
CUDA_VISIBLE_DEVICES=1 python continual_train.py --logs-dir logs/label-0.5 --label_ratio 0.5  
