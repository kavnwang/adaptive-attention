# Memory Routing Analysis

This directory contains analyses of memory routing mechanisms, particularly for models using MLP routers to select memory slots.

## Directory Structure

```
memory_routing/
├── scripts/
│   ├── analyze_gsm8k_routing.py     # Analyze routing on GSM8K dataset
│   ├── convert_memory_mlp_router.py  # Convert router checkpoints
│   └── calculate_active_parameters.py # Calculate active parameter usage
├── results/
│   ├── memory_mlp_router_340M_gsm8k_routing_with_positions/
│   └── routing_analysis_results/
└── visualizations/                   # Routing pattern visualizations
```

## Key Analyses

### Routing Pattern Analysis
- Token-level routing decisions
- Position-wise routing preferences
- Category-based routing patterns

### Active Parameter Calculation
- Measure parameter efficiency
- Compare active vs. total parameters
- Analyze sparsity patterns

### GSM8K Routing Study
- How mathematical reasoning uses memory
- Token category routing patterns
- Layer-wise routing distribution

## Visualization Types

- **Routing heatmaps**: Show which memory slots are selected
- **Position heatmaps**: Routing patterns by token position
- **Category heatmaps**: Routing by token type (numbers, operators, etc.)
- **Layer percentages**: Distribution across layers

## Running Analyses

1. Convert checkpoint if needed:
   ```bash
   python scripts/convert_memory_mlp_router.py --checkpoint exp/memory_mlp_router_340M
   ```

2. Analyze routing patterns:
   ```bash
   python scripts/analyze_gsm8k_routing.py --model_path exp/memory_mlp_router_340M
   ```

3. Calculate active parameters:
   ```bash
   python scripts/calculate_active_parameters.py --config memory_mlp_router_340M.json
   ```

## Key Metrics

- **Routing entropy**: Diversity of memory slot usage
- **Specialization**: Whether certain slots specialize for token types
- **Position bias**: If routing depends on token position
- **Layer utilization**: How different layers are used
