#!/bin/bash
#SBATCH --job-name=extract_memory_key_activations
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --time=2:00:00
#SBATCH --output=logs/extract_memory_key_activations_%j.log
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

# Run the key activation extraction script
# Using 8 GPUs (same as training configuration)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun python extract_memory_key_activations.py
