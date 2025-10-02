# Digit Lookup Experiments

This directory contains experiments for studying key-value memorization using digit lookup tasks.

## Dataset Format

The digit lookup dataset consists of key-value pairs where:
- Keys and values are 6-digit numbers
- Format: `key:value ` (space-terminated)
- Special test sets for known vs. novel pairs

## Directory Structure

```
digit_lookup/
├── datasets/
│   ├── synthetic_digit_lookup_data_6000/
│   ├── synthetic_digit_lookup_data_20000/
│   ├── synthetic_digit_lookup_data_200000/
│   └── *_special_tests/          # Known/novel pair test sets
├── scripts/
│   ├── generation/               # Dataset generation
│   └── evaluation/               # Model evaluation
└── results/                      # Experiment outputs
```

## Key Scripts

### Generation
- `generate_synthetic_digit_lookup_dataset.py` - Create digit lookup datasets

### Evaluation
- `eval_synthetic_digit_lookup.py` - Evaluate model on digit lookup tasks

## Dataset Sizes

- 6K pairs: Small-scale experiments
- 20K pairs: Medium-scale experiments  
- 200K pairs: Large-scale experiments

Each dataset includes:
- Train/validation/test splits
- Special test sets for generalization analysis

## Running Experiments

1. Generate dataset:
   ```bash
   python scripts/generation/generate_synthetic_digit_lookup_dataset.py --num_pairs 20000
   ```

2. Train model (use SLURM):
   ```bash
   sbatch llmonade/scripts/slurm/memory/digit_lookup/memory_2layer_digit_lookup_20K.slurm
   ```

3. Evaluate:
   ```bash
   python scripts/evaluation/eval_synthetic_digit_lookup.py --checkpoint exp/memory_2layer_digit_lookup_20K
   ```

## Configuration Files

Model configs: `llmonade/configs/memory/digit_lookup/`
SLURM jobs: `llmonade/scripts/slurm/memory/digit_lookup/`
