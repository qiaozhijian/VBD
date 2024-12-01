#!/bin/bash
#SBATCH --job-name=CacheData    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # number of processes
#SBATCH --mem=64G               # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:59:59         # total run time limit (HH:MM:SS)
#SBATCH --output=/n/fs/vbd/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/n/fs/vbd/slurm_output/%x.%j.err  # STDERR file

module purge
module load anaconda3/2024.02 

source /u/zixuz/.bashrc
conda activate vbd

srun python /n/fs/vbd/diffusion_waymax/script/extract_dataset.py \
    --data_dir /n/fs/vbd/data \
    --save_dir /n/fs/vbd/data_extract \
    --dataset val_interactive \
    --num_workers 16 \
    --only_raw   