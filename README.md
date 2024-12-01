# VBD - Versatile Behavior Diffusion
This repository contains the source code for the following paper:


[**Versatile Behavior Diffusion for Generalized Traffic Simulation**](https://arxiv.org/abs/2404.02524)

[Zhiyu Huang](https://mczhi.github.io/)<sup>1,\*</sup>, [Zixu Zhang](https://zzx9636.github.io/)<sup>2,\*</sup>, [Ameya Vaidya](https://scholar.google.com/citations?user=TWo54ggAAAAJ&hl=en)<sup>2</sup>, [Yuxiao Chen](https://research.nvidia.com/person/yuxiao-chen)<sup>3</sup>, [Chen Lv](https://lvchen.wixsite.com/automan)<sup>1</sup>, and [Jaime Fernández Fisac](https://ece.princeton.edu/people/jaime-fernandez-fisac)<sup>2</sup>

<sup>1</sup> Nanyang Technological University, <sup>2</sup> Princeton University, <sup>3</sup> NVIDIA Research


## Build Enviroment  
```bash
conda env create -n vbd -f environment.yml
conda activate vbd 
# Install waymax from source
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
pip install -e .
```

## Get Dataset
Download the Waymo Open Motion Dataset from [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/). Please use version V1.2 to work with Waymax.

## Data Preparation
Preprocess the data by running the following command:
```bash
python script/extract_data.py \
    --data_dir /path/to/waymo_open_motion_dataset \
    --save_dir /path/to/save_dir \
    --num_workers 16 \
    --save_raw # Extract Waymax Scenario for visualization
```

## Examples

- [Unguided Scenario with extracted data](example/unguided_generation.ipynb)
- [Guided Scenario with extracted data](example/goal_guided_generation.ipynb)



