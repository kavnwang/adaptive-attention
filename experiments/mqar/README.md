# MQAR (Multi-Query Associative Recall) Experiments

This directory contains experiments and utilities for the MQAR synthetic task, which tests language models' ability to perform associative recall.

## Task Description

MQAR is a synthetic task where models must learn to recall value tokens associated with key tokens. Each example consists of:

1. **Storage Phase**: `num_kv_pairs` key-value pairs presented sequentially
2. **Query Phase**: The same `num_kv_pairs` keys are queried in an order determined by a power law distribution

### Example Structure

For `num_kv_pairs=4`:
```
[K1 V1 K2 V2 K3 V3 K4 V4 K2 V2 K4 V4 K1 V1 K3 V3]
 |--------Storage--------|  |-------Queries-------|
```

The model is trained to predict values when given their corresponding keys during the query phase.

## Dataset Generation

### Basic Usage

Generate a MQAR dataset with default parameters:

```bash
python experiments/mqar/scripts/generation/generate_synthetic_mqar_dataset.py
```

### Advanced Usage

```bash
python experiments/mqar/scripts/generation/generate_synthetic_mqar_dataset.py \
    --output-dir ./mqar_data \
    --num-samples 100000 \
    --seq-len 512 \
    --num-kv-pairs 16 \
    --vocab-size 32000 \
    --power-a 0.01 \
    --num-copies 100 \
    --output-format parquet
```

### Parameters

- `--num-samples`: Number of unique sequences to generate per split
- `--seq-len`: Length of each sequence (should be divisible by 4×num_kv_pairs)
- `--num-kv-pairs`: Number of key-value pairs per MQAR example
- `--vocab-size`: Total vocabulary size (split evenly between keys and values)
- `--power-a`: Power law parameter controlling query order (smaller = prefer recent keys)
- `--num-copies`: Number of shuffled copies to concatenate (increases dataset size)
- `--output-format`: Choose between `parquet` (recommended) or `jsonl`

## Power Law Distribution

The `power_a` parameter controls how queries are ordered:
- `power_a = 0.01`: Strong preference for querying recently seen keys
- `power_a = 1.0`: More uniform distribution over all keys

The probability of querying a key at distance `d` from the end of storage is proportional to `d^(power_a - 1)`.

## Integration with LLMonade

The generated dataset is compatible with LLMonade's training pipeline:

```bash
# Use as synthetic dataset for initial training steps
--training.synthetic_dataset /path/to/mqar_data \
--training.synthetic_start_step 0 \
--training.synthetic_end_step 200
```

## Dataset Format

Each example contains:
- `input_ids`: Token sequence for training
- `labels`: Target tokens (-100 for non-prediction positions)
- `attention_mask`: All 1s (full attention)

The dataset follows the standard train/validation/test split structure with metadata tracking all generation parameters.

## Evaluation

MQAR performance is typically measured by:
- Accuracy on predicting values given keys
- Performance degradation with increasing `num_kv_pairs`
- Comparison across different model architectures