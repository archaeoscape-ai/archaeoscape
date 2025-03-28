<div align="center">

# Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era

</div>

[![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

## Description

This code includes the dataloader for the [archaeoscape dataset]() as well as the code needed to reproduce all experiments in the paper.

![Dataset overview](images/dataset_overview.png)
**The archaeoscape dataset** is derived from 888 km2 of aerial laser scans taken in Cambodia.
The 3D point cloud LiDAR data (left) was processed to obtain a digital terrain model (middle). 31,411 individual
polygons have been drawn and field-verified by archaeologists, delineating anthropogenic features (right)
## Installation

#### Conda

```bash
# clone project
git clone https://github.com/archaeoscape-ai/archaeoscape
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```
### Download model pretrained on ImageNet
A script is given(`scripts/download_pretrained.sh`) to automatize the downloading of all model finetuned in the benchmark.
```bash
# make script executable
chmod +x scripts/download_pretrained.sh

run script
scripts/download_pretrained.sh
```

Note: Larger model not use in the benchmark are commented out in the script. For all of them an associated config file is also given.

## How to run
This code make extensive use of hydra functionality. See [hydra documentation](https://hydra.cc/docs/intro/) for more information on hydra.


Train default model with default configuration (ViT small)

```bash
python src/train.py
```

Train Unet with default configuration

```bash
python src/train.py model=Unet
```

The list of all commands used for the experiments in the paper can be found in `scripts/benchmark.sh`. Some of them make use of the [hydra multirun functionality](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

## Pretrained models

Coming soon

## Acknowledgement
### Template
This code is based on [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### Funding
The experiments conducted in this study were performed using HPC/AI resources provided by GENCI-IDRIS (Grant 2023-AD011014781).

This work has made use of results obtained with the Chalawan HPC cluster, operated and maintained by the National Astronomical Research Institute of Thailand (NARIT) under the Ministry of Science and Technology of Royal Thai government.

This project is funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No 866454).
