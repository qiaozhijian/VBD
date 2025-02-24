# VBD - Versatile Behavior Diffusion

This repository contains the source code for the following paper:

[**Versatile Behavior Diffusion for Generalized Traffic Agent Simulation**](https://arxiv.org/abs/2404.02524)

[Zhiyu Huang](https://mczhi.github.io/)`<sup>`1,\*`</sup>`, [Zixu Zhang](https://zzx9636.github.io/)`<sup>`2,\*`</sup>`, [Ameya Vaidya](https://scholar.google.com/citations?user=TWo54ggAAAAJ&hl=en)`<sup>`2`</sup>`, [Yuxiao Chen](https://research.nvidia.com/person/yuxiao-chen)`<sup>`3`</sup>`, [Chen Lv](https://lvchen.wixsite.com/automan)`<sup>`1`</sup>`, and [Jaime Fernández Fisac](https://ece.princeton.edu/people/jaime-fernandez-fisac)`<sup>`2`</sup>`

`<sup>`1`</sup>` Nanyang Technological University, `<sup>`2`</sup>` Princeton University, `<sup>`3`</sup>` NVIDIA Research

## Build Enviroment

To set up the required environment, execute the following commands:

```bash
conda env create -n vbd -f environment.yml
conda activate vbd 
# Install waymax from source
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
pip install -e .
```

## Get Dataset

Download the Waymo Open Motion Dataset from [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/). Please use version V1.2 tf_example data to work with Waymax.

## Data Preparation

Preprocess the dataset using the following command:

```bash
python -c "import torch;print(torch.version.cuda)"
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CUDA_VISIBLE_DEVICES=7 python script/extract_data.py \
    --data_dir /home/qzj/datasets/waymo_open_dataset_motion/uncompressed/tf_example/validation_interactive \
    --save_dir /home/qzj/datasets/waymo_open_dataset_motion/uncompressed/tf_example/validation_interactive_processed \
    --num_workers 16 \
    --save_raw # Extract Waymax Scenario for visualization
```

Make sure to process training and validation data separately.

## Training

To train the VBD model, use the following command:

```bash
python script/train.py --cfg config/VBD.yaml --num_gpus 8
```

Update the ``VBD.yaml`` configuration file with appropriate parameters, such as the paths to the training and validation datasets.

## Testing

Run the following command to test the model in closed-loop simulation:

```bash
python script/test.py --test_set /path/to/data --model_path ./train_log/VBD/model.pth --save_simulation
```

Ensure that both ``--test_set`` and ``--model_path`` parameters are provided. The ``test_set`` parameter should point to the raw tf_example data file.

## Examples

Explore the following examples for different use cases:

- [Unguided Scenario with extracted data](example/unguided_generation.ipynb)
- [Guided Scenario with extracted data](example/goal_guided_generation.ipynb)

## Citation

If you find our repo or our paper useful, please use the following citation:

```
@article{huang2024versatile,
  title={Versatile Behavior Diffusion for Generalized Traffic Agent Simulation},
  author={Huang, Zhiyu and Zhang, Zixu and Vaidya, Ameya and Chen, Yuxiao and Lv, Chen and Fisac, Jaime Fern{\'a}ndez},
  journal={arXiv preprint arXiv:2404.02524},
  year={2024}
}
```
