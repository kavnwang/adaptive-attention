# LLMonade Experiments

This directory contains all experiment-related files organized by experiment type. Each experiment has its own subdirectory with a consistent structure.

## Directory Structure

```
experiments/
├── synthetic_qa/         # Q&A memorization experiments
├── digit_lookup/         # Digit pair lookup experiments
├── two_hop/             # Two-hop reasoning experiments
├── hashhop/             # Hash-hop experiments
├── memory_ablation/      # Memory layer ablation studies
├── memory_routing/       # MLP router and memory routing analysis
├── attention_analysis/   # Attention visualization and analysis
├── utils/               # Shared utilities and scripts
└── archive/             # Old results and deprecated scripts
```

## Experiment Organization

Each experiment directory follows this structure:
- `datasets/` - Training, validation, and test data
- `scripts/` - Generation, evaluation, and analysis scripts
- `results/` - Experiment outputs and results
- `visualizations/` - Plots and visual analysis

## Configuration and Job Files

- Model configurations remain in `llmonade/configs/`
- SLURM job files remain in `llmonade/scripts/slurm/`
- Both are organized by experiment type in subdirectories

## Quick Navigation

- [Synthetic QA](synthetic_qa/README.md) - Memory recall experiments
- [Digit Lookup](digit_lookup/README.md) - Key-value pair memorization
- [Two-Hop](two_hop/README.md) - Multi-hop reasoning tasks
- [HashHop](hashhop/README.md) - Hash-based reasoning experiments
- [Memory Ablation](memory_ablation/README.md) - Component analysis studies
- [Memory Routing](memory_routing/README.md) - Routing mechanism analysis
- [Attention Analysis](attention_analysis/README.md) - Attention pattern studies
