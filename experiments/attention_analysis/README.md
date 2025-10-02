# Attention Analysis

This directory contains analyses and visualizations of attention patterns in memory-augmented language models.

## Directory Structure

```
attention_analysis/
├── scripts/                     # Analysis scripts
├── results/
│   └── attention_scores.json    # Extracted attention scores
└── visualizations/
    ├── attention_head_*.png     # Individual head visualizations
    └── attention_grid_visualization.png  # Grid overview
```

## Visualization Types

### Individual Attention Heads
- Heatmaps showing attention patterns for specific heads
- Files named `attention_head_N.png` where N is the head index

### Grid Visualization
- Overview of all attention heads in a grid layout
- Useful for comparing patterns across heads

## Key Analyses

1. **Attention Pattern Types**:
   - Local/causal attention
   - Global attention
   - Specialized patterns (e.g., attending to specific token types)

2. **Head Specialization**:
   - Which heads focus on recent tokens
   - Which heads attend to distant context
   - Heads that show structured patterns

## Running Attention Analysis

1. Extract attention scores from a model:
   ```bash
   python scripts/extract_attention_scores.py --checkpoint exp/model_checkpoint
   ```

2. Generate visualizations:
   ```bash
   python scripts/visualize_attention.py --scores results/attention_scores.json
   ```

## Interpreting Results

- **Diagonal patterns**: Local/causal attention
- **Vertical stripes**: Tokens that receive global attention
- **Horizontal stripes**: Tokens that attend broadly
- **Block patterns**: Structured attention (e.g., sentence-level)

## Notable Findings

Check individual head visualizations for:
- Head 2, 4, 9: Often show interesting specialized patterns
- Head 11-14: May show global attention patterns
- Head 19-20, 24, 26-27, 30-31: Check for unique structures
