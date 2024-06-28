#!/bin/bash


#Normalization
python src/train.py  train.py data=new data.layers_names=["rgb","dtm"] model=Unet model.net.num_channels=4
python src/train.py  train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model=Unet model.net.num_channels=4
python src/train.py  train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=local model=Unet model.net.num_channels=4

#Main experiments
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 model=Unet,DLv3
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 model=ViT_S,HViT_S,Swin_S,PCPVT_S,PVTv2_S
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 model=ViT_dino_S
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 model=ViT_clipAI_B,ViT_clipLAION_B
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 model=ScaleViT_L_pre
#Bigger input image
python src/train.py --multirun train.py data=new data.layers_names=["rgb","dtm"] data.mean_type=["global","global","global","local"] model.net.num_channels=4 data.imagesize=512 data.imageside=256 model=Unet,ViT_S

#RGB only
python src/train.py  train.py data=new data.layers_names=["rgb"] data.mean_type=global model.net.num_channels=3 model=Unet
python src/train.py  train.py data=new data.layers_names=["rgb"] data.mean_type=global model.net.num_channels=3 model=PVTv2_S
python src/train.py  train.py data=new data.layers_names=["rgb"] data.mean_type=global model.net.num_channels=3 model=ViT_dino_S
#Ele only
python src/train.py  train.py data=new data.layers_names=["dtm"] data.mean_type=local model.net.num_channels=1 model=Unet
python src/train.py  train.py data=new data.layers_names=["dtm"] data.mean_type=local model.net.num_channels=1 model=PVTv2_S
python src/train.py  train.py data=new data.layers_names=["dtm"] data.mean_type=local model.net.num_channels=1 model=ViT_dino_S
