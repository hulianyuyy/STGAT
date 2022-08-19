# STGAT
This repo holds the code for: Spatial Temporal Graph Attention Network for Skeleton-Based Action Recognition [pdf](https://arxiv.org/abs/2208.08599)


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

Change the config file depending on what you want.

    `python train_val_test/train.py --config ./config/your_dataset/your_config_file.yaml`

Train with decoupled modalities by changing the 'num_skip_frame'(None to 1 or 2) option and 'decouple_spatial'(False to True) option in config file and train again. 
    
Then combine the generated scores with: 

    `python train_val_test/ensemble.py`
     