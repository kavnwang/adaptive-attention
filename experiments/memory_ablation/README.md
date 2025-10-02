# Memory Ablation Studies

This directory contains ablation studies and analyses of memory layer components to understand their contribution to model performance.

## Directory Structure

```
memory_ablation/
├── scripts/
│   ├── ablate_ablation_*.py    # Various ablation analysis scripts
│   └── swap_memory_keys.py      # Memory key swapping experiments
├── results/
│   └── memory_*layer_*_ablation.json    # Ablation results
├── key_activations/             # Extracted memory key activations
└── visualizations/              # Ablation visualizations
```

## Key Experiments

### Memory Key Ablation
- Systematically ablate individual memory keys
- Measure impact on recall accuracy
- Identify critical vs. redundant keys

### Attention Head Ablation
- Ablate attention heads in memory layers
- Analyze contribution to memory retrieval

### Memory Key Swapping
- Swap memory keys between positions
- Test if memory content is position-dependent

### Key Intervention Studies
- Intervene on specific memory keys
- Test causal relationships

## Visualization Types

- Ablation impact heatmaps
- Position-wise ablation effects
- Swap matrices showing key interactions
- Key intervention results

## Running Ablation Studies

1. Extract memory key activations:
   ```bash
   sbatch llmonade/scripts/slurm/memory/ablation/extract_memory_keys_200.slurm
   ```

2. Run ablation analysis:
   ```bash
   python scripts/ablate_memory_keys.py --checkpoint exp/memory_1layer_2K_no_mlp
   ```

3. Generate visualizations:
   ```bash
   python scripts/analyze_ablation_impact.py --results results/
   ```

## Key Findings Location

- Individual ablation results: `results/`
- Aggregated analyses: `visualizations/`
- Raw key activations: `key_activations/`
