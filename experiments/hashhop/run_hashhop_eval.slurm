#!/bin/bash
#SBATCH --job-name=hashhop_eval_340m
#SBATCH --output=logs/hashhop_eval_%j.out
#SBATCH --error=logs/hashhop_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load modules if needed (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate environment
source /home/kevin/LLMonade_interp/.venv/bin/activate

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create log directory
mkdir -p logs

# Model configuration
MODEL_PATH="exp/memory_mlp_router_340M_24K"
TOKENIZER_PATH="$MODEL_PATH"
DEVICE="cuda"

# Create results directory
RESULTS_DIR="hashhop_results_340m_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Starting Hash-Hop evaluation for model: $MODEL_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "="*60

# Function to run evaluation and capture metrics
run_eval() {
    local dataset=$1
    local output_dir="$RESULTS_DIR/$dataset"

    echo -e "\n\n--- Evaluating $dataset ---"
    echo "Time: $(date)"

    python analyze_hashhop_routing.py \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --dataset_path "hashhop_eval_340m/${dataset}.json" \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        2>&1 | tee "${output_dir}_log.txt"

    # Extract and display key metrics
    if [ -f "$output_dir/hashhop_results.json" ]; then
        echo "Extracting metrics for $dataset..."
        python -c "
import json
with open('$output_dir/hashhop_results.json', 'r') as f:
    data = json.load(f)
    m = data['overall_metrics']
    print(f'  Accuracy: {m[\"accuracy\"]:.2%} ({m[\"total_correct\"]}/{m[\"total_queries\"]})')
"
    fi
}

# Run evaluations in order of difficulty
echo -e "\n=== Starting evaluations ==="

# Easy dataset first
run_eval "easy_1hop"

# Medium difficulty
run_eval "medium_2hop"
run_eval "medium_2hop_cot"

# Hard difficulty
run_eval "hard_3hop"
run_eval "hard_3hop_cot"

# Challenge dataset (may take longer)
run_eval "challenge_4hop"

# Generate final summary report
echo -e "\n\n=== Generating Summary Report ==="
python -c "
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Collect all results
results = {}
for result_file in glob.glob('$RESULTS_DIR/*/hashhop_results.json'):
    dataset_name = Path(result_file).parent.name
    with open(result_file, 'r') as f:
        data = json.load(f)
    results[dataset_name] = data['overall_metrics']

# Create summary
summary = {
    'job_id': '$SLURM_JOB_ID',
    'model_path': '$MODEL_PATH',
    'results_dir': '$RESULTS_DIR',
    'datasets': results
}

# Save summary
with open('$RESULTS_DIR/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary table
print('\\n' + '='*70)
print('HASH-HOP EVALUATION SUMMARY')
print('='*70)
print(f'{'Dataset':20s} | {'Accuracy':>8s} | {'Correct':>12s} | {'Samples':>8s}')
print('-'*70)

for dataset in ['easy_1hop', 'medium_2hop', 'hard_3hop', 'challenge_4hop', 'medium_2hop_cot', 'hard_3hop_cot']:
    if dataset in results:
        m = results[dataset]
        print(f'{dataset:20s} | {m[\"accuracy\"]:8.2%} | {m[\"total_correct\"]:>5d}/{m[\"total_queries\"]:<6d} | {m[\"num_samples\"]:8d}')

# Create accuracy plot
datasets_ordered = ['easy_1hop', 'medium_2hop', 'hard_3hop', 'challenge_4hop']
cot_datasets = ['medium_2hop_cot', 'hard_3hop_cot']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Standard datasets
x1 = []
y1 = []
labels1 = []
for i, ds in enumerate(datasets_ordered):
    if ds in results:
        x1.append(i)
        y1.append(results[ds]['accuracy'] * 100)
        labels1.append(ds.replace('_', '\\\\n'))

ax1.bar(x1, y1, color='skyblue', edgecolor='navy', linewidth=2)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Hash-Hop Performance by Difficulty')
ax1.set_xticks(x1)
ax1.set_xticklabels(labels1)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in zip(x1, y1):
    ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

# CoT comparison
if 'medium_2hop' in results and 'medium_2hop_cot' in results:
    cot_data = [
        ['Medium\\\\n2-hop', results['medium_2hop']['accuracy']*100, results['medium_2hop_cot']['accuracy']*100],
    ]
    if 'hard_3hop' in results and 'hard_3hop_cot' in results:
        cot_data.append(['Hard\\\\n3-hop', results['hard_3hop']['accuracy']*100, results['hard_3hop_cot']['accuracy']*100])

    x2 = np.arange(len(cot_data))
    width = 0.35

    no_cot = [d[1] for d in cot_data]
    with_cot = [d[2] for d in cot_data]
    labels2 = [d[0] for d in cot_data]

    ax2.bar(x2 - width/2, no_cot, width, label='No CoT', color='lightcoral')
    ax2.bar(x2 + width/2, with_cot, width, label='With CoT', color='lightgreen')

    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Chain-of-Thought Impact')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('$RESULTS_DIR/summary_accuracy.png', dpi=300, bbox_inches='tight')
print(f'\\\\nSummary plot saved to: $RESULTS_DIR/summary_accuracy.png')
"

# Copy logs to results directory
cp "logs/hashhop_eval_${SLURM_JOB_ID}.out" "$RESULTS_DIR/"
cp "logs/hashhop_eval_${SLURM_JOB_ID}.err" "$RESULTS_DIR/"

echo -e "\n\n=== Evaluation Complete ==="
echo "Time: $(date)"
echo "Results directory: $RESULTS_DIR"
echo "Job completed successfully!"

# Optional: sync to permanent storage if needed
# rsync -av "$RESULTS_DIR" /path/to/permanent/storage/
