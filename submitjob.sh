#!/bin/bash

#SBATCH --job-name=submitjob
#SBATCH --output=logs/%j.out
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00

module load brics/nccl
module load brics/aws-ofi-nccl
module load cuda/12.6
module load gcc-native/12.3

export CONDA_ENV=/scratch/u6if/yangw.u6if/miniforge3/bin/conda/envs/gliner2
export PATH=$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1

sleep infinity