# Two-Hop Experiments

This directory contains experiments for studying multi-hop reasoning in memory-augmented language models.

## Dataset Format

The two-hop dataset tests the model's ability to perform transitive reasoning:
- First hop: A → B
- Second hop: B → C
- Query: A → ?
- Answer: C

## Directory Structure

```
two_hop/
├── datasets/
│   ├── two_hop_data/            # Main dataset
│   └── two_hop_data_special_tests/
│       ├── test_known_pairs/    # Direct A→C pairs seen in training
│       └── test_novel_two_hop/  # Novel A→C requiring reasoning
├── scripts/
│   └── generation/              # Dataset generation
└── results/                     # Experiment outputs
```

## Key Scripts

### Generation
- `generate_synthetic_two_hop_dataset.py` - Create two-hop reasoning datasets

## Running Experiments

1. Generate dataset:
   ```bash
   python scripts/generation/generate_synthetic_two_hop_dataset.py
   ```

2. Train model (use SLURM):
   ```bash
   sbatch llmonade/scripts/slurm/memory/two_hop/memory_2layer_two_hop_1M.slurm
   ```

## Special Test Sets

- **Known pairs**: Tests memorization of directly seen A→C mappings
- **Novel two-hop**: Tests true multi-hop reasoning on unseen A→C pairs

## Configuration Files

Model configs: `llmonade/configs/memory/two_hop/`
SLURM jobs: `llmonade/scripts/slurm/memory/two_hop/`
