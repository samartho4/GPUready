# Scripts Directory

This directory contains the essential scripts for the Microgrid Bayesian Neural ODE Control project.

## Core Scripts

### `focused_ude_bnode_evaluation.jl`
**Main evaluation script** - Comprehensive comparison of UDE vs BNODE approaches
- Data quality assessment
- Model architecture analysis
- Training efficiency comparison
- Predictive performance evaluation
- Practical recommendations

### `comprehensive_ude_bnode_evaluation.jl`
**Extended evaluation script** - More detailed analysis with additional metrics
- Cross-validation
- Statistical significance testing
- Robustness analysis
- Comprehensive diagnostics

### `train_ude_optimization.jl`
**UDE training script** - Optimized Universal Differential Equation training
- L-BFGS optimization
- Point estimation
- Performance metrics
- Model checkpointing

## Usage

```bash
# Run main evaluation
julia scripts/focused_ude_bnode_evaluation.jl

# Run comprehensive evaluation
julia scripts/comprehensive_ude_bnode_evaluation.jl

# Train UDE model
julia scripts/train_ude_optimization.jl
```

## Key Findings

- **UDE recommended** for current application (25x faster, simpler)
- **Data quality** is primary limiting factor
- **BNODE provides uncertainty** at computational cost
- **Model choice** depends on application requirements

## Archive

Redundant and experimental scripts have been moved to `archive/redundant_scripts/` for reference. 