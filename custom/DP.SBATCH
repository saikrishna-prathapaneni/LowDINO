#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=64GB           # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4
#SBATCH --job-name=torch
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=saiprathapaneni@nyu.edu

module purge

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

singularity exec --nv \
            --overlay /scratch/sp7238/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python train_dp.py \
            "
