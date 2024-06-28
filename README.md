<div align="center">

# Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era

</div>

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
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

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

The list of all commands used for the experiments in the paper can be found in `scripts/archeoscape.sh`. Some of them make use of the [hydra multirun functionality](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

## Pretrained models

Coming soon

## Acknowledgement
### Template
This code is based on [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### Funding
The experiments conducted in this study were performed using HPC/AI resources provided by GENCI-IDRIS (Grant 2023-AD011014781).

This work has made use of results obtained with the Chalawan HPC cluster, operated and maintained by the National Astronomical Research Institute of Thailand (NARIT) under the Ministry of Science and Technology of Royal Thai government.

This project is funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No 866454).
