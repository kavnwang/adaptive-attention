# Synthetic QA Experiments

This directory contains experiments for studying memorization and recall in memory-augmented language models using synthetic question-answering datasets.

## Dataset Format

The synthetic QA dataset consists of entity pairs (6-digit numbers) in the format:
- Forward recall: `entity_x:entity_y ` (space-terminated)
- Each pair appears with controlled frequency (1, 2, 4, 8, 16, 32, 64, 128 times)

## Directory Structure

```
synthetic_qa/
├── datasets/
│   ├── synthetic_qa_data_200/    # 200 unique pairs
│   ├── synthetic_qa_data_2K/     # 2,000 unique pairs
│   ├── synthetic_qa_data_10K/    # 10,000 unique pairs
│   └── synthetic_qa_data_50K/    # 50,000 unique pairs
├── scripts/
│   ├── generation/              # Dataset generation scripts
│   ├── evaluation/              # Model evaluation scripts
│   ├── analysis/                # Analysis and visualization
│   └── ablation/                # Ablation study scripts
└── results/
    └── ablation_results/        # Ablation study outputs
```

## Key Scripts

### Generation
- `generate_synthetic_qa_dataset.py` - Create synthetic QA datasets

### Evaluation
- `eval_synthetic_qa_simple.py` - Basic evaluation script
- `eval_synthetic_qa_logits.py` - Logit-based analysis
- `extract_memory_key_activations.py` - Extract memory layer activations

### Ablation Studies
- `ablate_memory_keys.py` - Memory key ablation experiments
- `ablate_attention_heads.py` - Attention head ablation
- `intervene_memory_keys.py` - Memory key intervention studies

## Running Experiments

1. Generate dataset:
   ```bash
   python scripts/generation/generate_synthetic_qa_dataset.py --num_pairs 10000
   ```

2. Train model (use SLURM):
   ```bash
   sbatch llmonade/scripts/slurm/memory/synthetic_qa/memory_2layer_synthetic_qa_10K.slurm
   ```

3. Evaluate:
   ```bash
   python scripts/evaluation/eval_synthetic_qa_simple.py --checkpoint exp/memory_2layer_synthetic_qa_10K
   ```

## Configuration Files

Model configs: `llmonade/configs/memory/synthetic_qa/`
SLURM jobs: `llmonade/scripts/slurm/memory/synthetic_qa/`
