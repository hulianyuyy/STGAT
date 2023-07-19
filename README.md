# STGAT
This repo holds the code for: Skeleton-Based Action Recognition with Local Dynamic Spatial-Temporal Aggregation ([pdf](https://authors.elsevier.com/a/1hL6Z3PiGTPe0F)) (Previous name: Spatial Temporal Graph Attention Network for Skeleton-Based Action Recognition([pdf](https://arxiv.org/abs/2208.08599)))


# Data Preparation

 - NTU-60
    - Download the NTU-60 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_60/gendata.py`
 - NTU-120
    - Download the NTU-120 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_120/gendata.py`
 - Knietics-400
    - Download the data from ST-GCN repo: https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton
    - Generate the train/test splits with `python prepare/kinetics_gendata.py`
    
     
# Training & Testing
Change line 10 in train.py to the absolute path of this STGAT repo you download.

Set the data_path and label_path to the path of your dataset in line 4,5,13,14, and specify the cuda_visible_device & device_id in line 61, 62, of /train_val_test/config/your_dataset/your_config_file.yaml, respectively.

Change the config file depending on what you want.

    `python train_val_test/train.py -config ./train_val_test/config/your_dataset/your_config_file.yaml`

Train with decoupled modalities by changing the 'num_skip_frame'(None to 1 or 2) option and 'decouple_spatial'(False to True) option in config file and train again. 
    
Then combine the generated scores with: 

    `python train_val_test/ensemble.py`
     