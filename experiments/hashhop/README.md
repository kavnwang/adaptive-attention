# HashHop Experiments

This directory contains experiments using the HashHop benchmark for testing multi-hop reasoning with hash-based entity mappings.

## About HashHop

HashHop is a benchmark that tests models' ability to perform multi-hop reasoning through hash function compositions. It includes tasks of varying difficulty from 1-hop to 4-hop reasoning.

## Directory Structure

```
hashhop/
├── datasets/
│   ├── hashhop_1hop_training/      # 1-hop training data
│   ├── hashhop_1hop_training_parquet/
│   └── hashhop_eval_340m/          # Evaluation sets
│       ├── easy_1hop.json
│       ├── medium_2hop.json
│       ├── hard_3hop.json
│       └── challenge_4hop.json
├── scripts/                         # Evaluation and analysis scripts
├── results/
│   └── hashhop_results_340m/       # Model evaluation results
├── hash-hop/                        # Original HashHop repository
└── *.slurm                         # SLURM job files
```

## Key Scripts

### Dataset Creation
- `create_hashhop_1hop_training.py` - Generate 1-hop training data
- `create_hashhop_eval.py` - Create evaluation sets

### Evaluation
- `run_hashhop_eval.py` - Run full HashHop evaluation
- `eval_hashhop_example.py` - Example evaluation script
- `analyze_hashhop_routing.py` - Analyze memory routing patterns

## Running Experiments

1. Create training data:
   ```bash
   python scripts/create_hashhop_1hop_training.py
   ```

2. Train model:
   ```bash
   sbatch memory_340M_hashhop.slurm
   ```

3. Evaluate:
   ```bash
   python scripts/run_hashhop_eval.py --checkpoint exp/memory_340M_hashhop
   ```

## Evaluation Sets

- **Easy (1-hop)**: Direct hash mapping
- **Medium (2-hop)**: Two-step reasoning with CoT
- **Hard (3-hop)**: Three-step reasoning with CoT
- **Challenge (4-hop)**: Four-step reasoning

## Results Analysis

Results include:
- Accuracy metrics for each difficulty level
- Memory routing patterns
- Category-wise performance analysis
