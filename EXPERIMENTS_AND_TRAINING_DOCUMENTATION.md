# LLMonade Experiments and Training Pipeline Documentation

This document provides comprehensive documentation for all experiment files and training pipeline modifications in the LLMonade repository.

## Table of Contents

1. [Experiments Documentation](#experiments-documentation)
   - [Synthetic QA Experiments](#synthetic-qa-experiments)
   - [Digit Lookup Experiments](#digit-lookup-experiments)
   - [Two-Hop Experiments](#two-hop-experiments)
   - [HashHop Experiments](#hashhop-experiments)
   - [Memory Ablation Experiments](#memory-ablation-experiments)
   - [Memory Routing Experiments](#memory-routing-experiments)
   - [Attention Analysis](#attention-analysis)
   - [Utility Scripts](#utility-scripts)
2. [Training Pipeline Modifications](#training-pipeline-modifications)
   - [Component Freezing Infrastructure](#component-freezing-infrastructure)
   - [Pretrained Layer Loading Infrastructure](#pretrained-layer-loading-infrastructure)
   - [Masking Framework](#masking-framework)
3. [File Organization Changes](#file-organization-changes)

---

## Experiments Documentation

### Synthetic QA Experiments

Location: `experiments/synthetic_qa/`

#### Generation Scripts

**`generate_synthetic_qa_dataset.py`**
- **Purpose**: Generates synthetic question-answering datasets for memorization studies
- **Key Features**:
  - Creates entity pairs (6-digit numbers) with controlled exposure frequencies
  - Supports frequency buckets: 1, 2, 4, 8, 16, 32, 64, 128 repetitions
  - Train/val/test splits with held-out items in each frequency bucket
  - Output format: "entity_x:entity_y " (space-terminated)
- **Usage**:
  ```bash
  python generate_synthetic_qa_dataset.py --num_pairs 10000 --output_dir synthetic_qa_data_10K
  ```

#### Evaluation Scripts

**`eval_synthetic_qa_simple.py`**
- **Purpose**: Basic evaluation of models on synthetic QA tasks
- **Method**: Forward pass evaluation with exact match accuracy
- **Output**: JSON file with per-frequency accuracy metrics

**`eval_synthetic_qa_logits.py`**
- **Purpose**: Detailed logit-level analysis of model predictions
- **Features**:
  - Extracts logit distributions for answer tokens
  - Analyzes confidence scores and prediction probabilities
  - Useful for understanding model uncertainty

**`eval_synthetic_qa_fast.py`**
- **Purpose**: Optimized evaluation for large-scale experiments
- **Features**: Batch processing, GPU optimization, parallel evaluation

**`extract_memory_key_activations.py`**
- **Purpose**: Extracts memory layer key activations during inference
- **Method**: Hooks into memory retrieval to capture key-query similarities
- **Output**: JSON file with activation patterns for each example

**`extract_value_vector_norms.py`**
- **Purpose**: Analyzes the magnitude of memory value vectors
- **Use Case**: Understanding which memory slots store stronger representations

**`track_memory_activations.py`**
- **Purpose**: Real-time tracking of memory usage during training
- **Features**: Logs memory access patterns, slot utilization statistics

**`run_synthetic_qa_eval.py`**
- **Purpose**: Orchestrates full evaluation pipeline
- **Features**: Runs multiple evaluation scripts in sequence

#### Ablation Scripts

**`ablate_memory_keys.py`**
- **Purpose**: Systematically removes memory keys to measure impact
- **Method**:
  - Iterates through each memory key
  - Sets key to zero or random values
  - Measures drop in recall accuracy
- **Output**: Ablation impact matrix showing importance of each key

**`ablate_memory_layer.py`**
- **Purpose**: Ablates entire memory layers or components
- **Variations**: Full layer removal, value-only ablation, key-only ablation

**`ablate_attention_heads.py`**
- **Purpose**: Identifies critical attention heads in memory layers
- **Method**: Individual head masking with performance measurement

**`intervene_memory_keys.py` and `intervene_memory_keys_v2.py`**
- **Purpose**: Causal intervention studies on memory keys
- **Methods**:
  - Key swapping between positions
  - Key replacement with specific patterns
  - Activation injection experiments

**`analyze_swap_matrix.py`**
- **Purpose**: Analyzes results from key swapping experiments
- **Output**: Visualization of which keys can substitute for others

**`analyze_swap_predictions.py`**
- **Purpose**: Detailed analysis of model predictions after interventions
- **Features**: Tracks how predictions change with different key configurations

**`analyze_ablation_impact.py`**
- **Purpose**: Aggregates and visualizes ablation results
- **Output**: Heatmaps and summary statistics

**`analyze_qa_key_statistics.py`**
- **Purpose**: Statistical analysis of memory key properties
- **Metrics**: Activation frequency, variance, clustering patterns

**`run_ablations_all.py`**
- **Purpose**: Batch execution of all ablation experiments
- **Features**: Parallel execution, result aggregation

#### Analysis Scripts

**`analyze_key_occurrences.py`**
- **Purpose**: Tracks how often each memory key is accessed
- **Use Case**: Identifying unused or redundant memory slots

**`analyze_memory_dimension_mismatch.py`**
- **Purpose**: Debugs dimension mismatches in memory operations
- **Features**: Shape validation, tensor dimension tracking

**`analyze_repeating_digits_correlation.py`**
- **Purpose**: Studies correlation between digit patterns and memory usage
- **Context**: Some 6-digit entities have repeating patterns (e.g., 111111)
- **Output**: Correlation analysis and visualizations

### Digit Lookup Experiments

Location: `experiments/digit_lookup/`

**`generate_synthetic_digit_lookup_dataset.py`**
- **Purpose**: Creates key-value pair datasets for lookup tasks
- **Format**: "key:value " where both are 6-digit numbers
- **Features**:
  - Controllable dataset sizes (6K, 20K, 200K pairs)
  - Special test sets for known vs. novel pairs
  - Tests pure memorization without reasoning

**`eval_synthetic_digit_lookup.py`**
- **Purpose**: Evaluates models on digit lookup tasks
- **Metrics**:
  - Exact match accuracy
  - Partial digit accuracy
  - Generalization to novel keys

### Two-Hop Experiments

Location: `experiments/two_hop/`

**`generate_synthetic_two_hop_dataset.py`**
- **Purpose**: Creates datasets for multi-hop reasoning evaluation
- **Task Structure**:
  - Training: A→B and B→C mappings
  - Test: Query A, expect C (requires two-hop reasoning)
- **Features**:
  - Known pairs test set (direct A→C seen in training)
  - Novel two-hop test set (requires actual reasoning)

### HashHop Experiments

Location: `experiments/hashhop/`

**`create_hashhop_1hop_training.py`**
- **Purpose**: Generates 1-hop hash mapping training data
- **Format**: Simple hash function mappings for initial training

**`create_hashhop_1hop_training_parquet.py`**
- **Purpose**: Converts HashHop data to Parquet format
- **Use Case**: Efficient data loading for large-scale training

**`create_hashhop_eval.py`**
- **Purpose**: Creates multi-hop evaluation sets
- **Difficulty Levels**:
  - Easy: 1-hop direct mappings
  - Medium: 2-hop with chain-of-thought
  - Hard: 3-hop reasoning
  - Challenge: 4-hop complex reasoning

**`create_hashhop_training_dataset.py`**
- **Purpose**: Full training dataset generation pipeline
- **Features**: Balanced difficulty distribution, diverse hash functions

**`run_hashhop_eval.py`**
- **Purpose**: Main evaluation script for HashHop benchmark
- **Features**:
  - Multi-GPU support
  - Detailed accuracy breakdowns by difficulty
  - Error analysis

**`eval_hashhop_example.py`**
- **Purpose**: Example/demo script for single instance evaluation
- **Use Case**: Debugging and understanding model behavior

**`test_hashhop_tokenization.py`**
- **Purpose**: Validates tokenization of hash values
- **Importance**: Ensures consistent token boundaries for hash strings

**`analyze_hashhop_routing.py`**
- **Purpose**: Analyzes memory routing patterns on HashHop tasks
- **Features**:
  - Which memory slots are used for hash lookups
  - Routing consistency across similar inputs
  - Layer-wise routing distribution

### Memory Ablation Experiments

Location: `experiments/memory_ablation/`

**`swap_memory_keys.py`**
- **Purpose**: Swaps memory keys between positions
- **Method**:
  - Systematic pairwise swapping
  - Measures impact on recall accuracy
  - Tests position-dependence of memory content

**`analyze_ablation_by_position.py`**
- **Purpose**: Analyzes ablation impact by token position
- **Insight**: Some positions may be more critical than others

**`analyze_ablation_swap_correlation.py`**
- **Purpose**: Correlation analysis of swap experiment results
- **Output**: Which keys are functionally similar/interchangeable

### Memory Routing Experiments

Location: `experiments/memory_routing/`

**`analyze_gsm8k_routing.py`**
- **Purpose**: Studies memory routing on math reasoning tasks (GSM8K)
- **Analysis**:
  - Token-type routing (numbers vs. operators vs. words)
  - Position-based routing patterns
  - Layer-wise routing distribution
- **Output**: Detailed routing heatmaps and statistics

**`convert_memory_mlp_router.py`**
- **Purpose**: Converts checkpoints between memory model formats
- **Use Case**: Compatibility between different memory architectures

**`calculate_active_parameters.py`**
- **Purpose**: Calculates effective parameter usage in sparse models
- **Metrics**:
  - Total parameters vs. active parameters
  - Sparsity patterns
  - Memory efficiency analysis

### Attention Analysis

Location: `experiments/attention_analysis/`

**Visualization Files**:
- `attention_head_*.png`: Individual attention head pattern visualizations
- `attention_grid_visualization.png`: Grid overview of all heads
- `attention_scores.json`: Raw attention score data

**Key Patterns Observed**:
- Head 2, 4, 9: Local/causal attention patterns
- Head 11-14: Global attention to specific tokens
- Head 19-20, 24, 26-27, 30-31: Specialized patterns for memory operations

### Utility Scripts

Location: `experiments/utils/`

**`debug_memory_checkpoint_loading.py`**
- **Purpose**: Debugging tool for checkpoint loading issues
- **Features**: Verbose logging, shape validation, state dict inspection

**`explain_statistics.py`**
- **Purpose**: Statistical analysis helper for experiment results
- **Features**: Significance testing, confidence intervals

**`fix_and_analyze_json.py`**
- **Purpose**: Repairs corrupted JSON output files
- **Use Case**: Recovering results from interrupted experiments

**`plot_loss_comparison.py`**
- **Purpose**: Creates comparison plots of training losses
- **Features**: Multi-run averaging, confidence bands

**`calculate_active_parameters.py`**
- **Purpose**: Parameter counting for sparse/dynamic models
- **Context**: Memory models have dynamic parameter usage

---

## Training Pipeline Modifications

### Component Freezing Infrastructure

Location: `llmonade/train.py`

**`freeze_model_layers()` Function**
```python
def freeze_model_layers(
    model_parts: List[torch.nn.Module],
    freeze_layer_map: Union[str, Dict[str, str]],
    strict: bool = False
) -> int:
```

**Features**:
1. **Pattern-Based Freezing**:
   - Direct match: `"model.layers.0.attn"` → Freezes exact parameter
   - Wildcard: `"*.memory.keys"` → Freezes all memory keys across layers
   - Prefix: `"model.layers.0."` → Freezes entire layer 0

2. **Configuration Format**:
   ```json
   {
     "model.layers.0.memory.*": "freeze",
     "*.attn.k_proj": "freeze",
     "model.layers.1": "freeze"
   }
   ```

3. **Integration**:
   - Called after model initialization but before optimizer creation
   - Sets `requires_grad=False` on matched parameters
   - Reports total frozen parameters

4. **Use Cases**:
   - Freezing attention while training memory layers
   - Freezing early layers for transfer learning
   - Component-specific fine-tuning

### Pretrained Layer Loading Infrastructure

Location: `llmonade/components/checkpoint.py`

**`PretrainedLayerLoader` Class**

**Key Methods**:

1. **Layer Mapping Strategies**:
   - **Index Mapping**: `{"0-5": "0-5", "6": "10"}` - Maps layer ranges
   - **Component Paths**: `{"model.layers.0.memory.values": "model.layers.1.memory.values"}`
   - **Pattern Matching**: `{"*.memory.keys": "*.memory.keys"}` - Preserves structure

2. **Distributed Loading**:
   - Handles sharded checkpoints across multiple GPUs
   - Converts distributed checkpoints to single file for processing
   - Re-shards for target model parallelism configuration

3. **Configuration Example**:
   ```json
   {
     "layer_mappings": {
       "0-5": "0-5"
     },
     "component_mappings": {
       "model.layers.0.memory.*": "model.layers.0.memory.*",
       "model.layers.0.mlp.*": "model.layers.1.mlp.*"
     }
   }
   ```

4. **SLURM Integration**:
   ```bash
   --checkpoint.pretrained_layer_path exp/memory_1layer_synthetic_qa_2K \
   --checkpoint.pretrained_layer_step 10000 \
   --checkpoint.pretrained_layer_map '{"component_mappings": {...}}'
   ```

### Masking Framework

Location: `llmonade/data.py`

**`DataCollatorForLanguageModelingVarLen` Class**

**Three Masking Types**:

1. **Synthetic QA Masking** (`apply_synthetic_qa_mask`):
   - **Pattern**: Masks everything before the last '?' token
   - **Purpose**: Train only on answer portions
   - **Example**:
     - Input: "What is the capital of France?"
     - Masked: "What is the capital of France?"
     - Unmasked: "Paris"

2. **HashHop Masking** (`apply_hashhop_mask`):
   - **Pattern**: Only unmasks hash values between quotes after '='
   - **Purpose**: Focus learning on hash mappings
   - **Example**:
     - Input: "KmNp = 'QrSt'"
     - Masked: "KmNp = '" and "'"
     - Unmasked: "QrSt"

3. **Digit Lookup Masking** (`apply_digit_lookup_mask`):
   - **Pattern**: Unmasks only value portions after ':'
   - **Purpose**: Pure lookup table learning
   - **Example**:
     - Input: "152435:145344 "
     - Masked: "152435:" and " "
     - Unmasked: "145344"

**Implementation Features**:
- Supports both 2D batch format and 1D varlen format
- Parallelizable across sequences
- Efficient token-based masking
- Compatible with FlashAttention varlen format

**Configuration**:
```python
# In dataloader creation
build_dataloader(
    ...
    apply_qa_mask=True,  # or apply_hashhop_mask, apply_digit_lookup_mask
)
```

---

## File Organization Changes

### Original Structure (Root Directory)
```
/
├── analyze_*.py                    # Scattered analysis scripts
├── generate_*.py                   # Dataset generation scripts
├── eval_*.py                       # Evaluation scripts
├── synthetic_qa_experiments/       # QA experiment directory
├── hashhop_experiments/           # HashHop directory
├── memory_mlp_router/             # Routing analysis
├── attention_visualizations/       # Attention plots
├── ablation_results/              # Ablation outputs
└── various result files
```

### New Structure
```
experiments/
├── synthetic_qa/
│   ├── datasets/                  # All QA datasets
│   ├── scripts/
│   │   ├── generation/           # Dataset creation
│   │   ├── evaluation/           # Model evaluation
│   │   ├── analysis/             # Result analysis
│   │   └── ablation/             # Ablation studies
│   └── results/                  # Experiment outputs
├── digit_lookup/                 # Similar structure
├── two_hop/                      # Similar structure
├── hashhop/                      # Similar structure
├── memory_ablation/              # Ablation-specific studies
├── memory_routing/               # Routing analysis
├── attention_analysis/           # Attention studies
├── utils/                        # Shared utilities
└── archive/                      # Old results

llmonade/
├── configs/
│   └── memory/
│       ├── synthetic_qa/         # QA-specific configs
│       ├── digit_lookup/         # Lookup configs
│       └── two_hop/              # Two-hop configs
└── scripts/
    └── slurm/
        └── memory/
            ├── synthetic_qa/     # QA SLURM files
            ├── digit_lookup/     # Lookup SLURM files
            ├── two_hop/          # Two-hop SLURM files
            └── ablation/         # Ablation SLURM files
```

### Benefits of Reorganization

1. **Clear Experiment Separation**: Each experiment type has its own space
2. **Consistent Structure**: All experiments follow the same organization pattern
3. **Easy Navigation**: Related files are grouped together
4. **Scalability**: Easy to add new experiment types
5. **Clean Root**: No scattered files in the repository root

### Migration Notes

- All Python scripts were moved with their imports intact
- SLURM files reference the new locations
- Configuration files were organized by experiment type
- Old results were archived to preserve experiment history
- README files were added to each directory for documentation
