#!/bin/bash
#SBATCH --job-name=test_key_ablation
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_key_ablation_%j.log
#SBATCH --exclude=worker-4,worker-5

# Important environment variables
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Set up Triton cache directory
export TRITON_CACHE_DIR=~/tmp/triton_cache_user_owned
mkdir -p $TRITON_CACHE_DIR
chmod 777 $TRITON_CACHE_DIR

# Run the key ablation testing script
# Using single GPU since this is inference/testing
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun python test_key_ablation.py
