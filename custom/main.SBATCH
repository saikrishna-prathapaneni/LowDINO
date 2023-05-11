#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=torch

module purge

singularity exec --nv \
            --overlay /scratch/sp7238/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python train.py"
