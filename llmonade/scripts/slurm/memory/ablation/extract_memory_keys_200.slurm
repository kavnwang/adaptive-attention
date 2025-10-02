#!/bin/bash
#SBATCH --job-name=extract_memory_keys_200
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/extract_memory_keys_200_%j.log

# Important environment variables
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Set up Triton cache directory
export TRITON_CACHE_DIR=~/tmp/triton_cache_user_owned
mkdir -p $TRITON_CACHE_DIR
chmod 777 $TRITON_CACHE_DIR

# Create logs and results directories
mkdir -p logs
mkdir -p results

# Run the key activation extraction script
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun python extract_memory_key_activations.py \
    --model_path exp/memory_2layer_synthetic_qa_200 \
    --dataset_path experiments/synthetic_qa/datasets/synthetic_qa_data_200 \
    --output_path results/memory_2layer_200_key_activations.json \
    --batch_size 32
