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

# Visualization 
To visualize the attention weights of the center frame, you could first (1) select an index of input samples, and only pass this sample to the training process by uncommenting line 51 in ./train_val_test/train_val_model.py and change 'xxxx' into the index you select. Note that you should indent line 52~line 101, and change the training batch size into 1 in the config file (e.g., line 51 in ./train_val_test/config/ntu/ntu60_dsta.yaml); (2) uncomment line 217 in ./model/st2ransformer.py. After that, you should train or test this code (with pretrained weights) on a dataset you desire. You could get a 'stgat_attention.npy' under your folder. Next, run the 'ntu_visualize_add_attention.py' with desired parameter. Note that the indices you pass into the py file should be consistent with that in the './train_val_test/train_val_model.py'. 

You may also change some hyperparameters in the files to change the visualization results (e.g., the threshold of line 209 in 'ntu_visualize_add_attention.py').
     