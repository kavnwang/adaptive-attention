# Sorting Experiment

This experiment evaluates whether models learn to memorize key-value pairs or learn the underlying sorting algorithm.

## Dataset Format

The dataset consists of key-value pairs where:
- **Keys**: N-digit numbers with digits 0-9
- **Values**: The sorted digits of the key
- **Format**: `"key:value "` (with trailing space)

Examples:
- `"43221:12234 "`
- `"16783:13678 "`
- `"43444:34444 "`

## Key Insights

By varying the number of unique key-value pairs (`num_keys`), we can determine:
1. At what scale the model switches from memorization to learning the sorting algorithm
2. Whether the model can generalize to unseen keys

## Directory Structure

```
sorting/
├── README.md
├── scripts/
│   ├── generation/         # Dataset generation scripts
│   └── evaluation/         # Model evaluation scripts
├── datasets/              # Generated datasets
├── results/              # Evaluation results
└── visualizations/       # Plots and analysis
```

## Usage

### Generate Dataset

```bash
python experiments/sorting/scripts/generation/generate_synthetic_sorting_dataset.py \
    --num-keys 1000 \
    --output-dir ./experiments/sorting/datasets/synthetic_sorting_data \
    --seed 42
```

### Evaluate Model

```bash
# TODO: Add evaluation script
```

## Experimental Design

The experiment tests different values of `num_keys` to find the transition point where:
- Small `num_keys`: Model memorizes all key-value pairs
- Large `num_keys`: Model learns the sorting algorithm

The minimum vocabulary size is automatically determined as 10^N where N is the smallest integer such that 10^N >= num_keys.
