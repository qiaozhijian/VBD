#!/bin/bash
#SBATCH --job-name=TrainVBD     # create a short name for your job
#SBATCH --nodes=1               # node count # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=8     # This needs to match Trainer(devices=...)
#SBATCH --gres=gpu:8            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of processes DataLoader(num_workers=...) 
#SBATCH --mem-per-cpu=2G        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=143:59:59        # total run time limit (HH:MM:SS)
#SBATCH --output=/n/fs/vbd/slurm_output/%x.%j.out   # STDOUT file
#SBATCH --error=/n/fs/vbd/slurm_output/%x.%j.err  # STDERR file
export SRUN_CPUS_PER_TASK=8

. /etc/profile

module purge
module load anaconda3/2024.02 

source /u/zixuz/.bashrc
conda activate vbd

wandb offline

srun python /n/fs/vbd/diffusion_waymax/script/train.py \
    --cfg /n/fs/vbd/diffusion_waymax/config/VBD_slurm.yaml
