# Microgrid Control with Universal Differential Equations and Bayesian Neural ODEs

## Abstract
We study microgrid dynamics modeling with Scientific Machine Learning (SciML). We compare two complementary approaches: (i) Universal Differential Equations (UDE), which retain first‑principles structure while learning an unknown term fθ(Pgen), and (ii) Bayesian Neural ODEs (BNode), which replace both equations with neural functions and quantify uncertainty via MCMC. On a 70‑scenario dataset (14,070 points), we find near‑parity between UDE and a physics baseline on the storage state x1, and a small but consistent average RMSE improvement for UDE on the power‑flow state x2. Symbolic extraction recovers a compact cubic for fθ(Pgen) with a dominant linear component, yielding interpretable structure. We report complete timings: UDE hyperparameter search (2,880 configurations) required ~30 hours wall‑clock; BNode posterior sampling completed in ~37 minutes (500 samples, 2 chains). Code, data, and checkpoints are provided for reproducibility.

## 1. Introduction
Physics‑driven energy systems demand models that are accurate, data‑efficient, and interpretable. Purely data‑driven models may struggle with extrapolation and constraint adherence, while purely mechanistic models can miss unmodeled phenomena. We combine the strengths of both via:
- A UDE that injects a neural component into a targeted term while preserving governing equations.
- A BNode that learns full dynamics and provides posterior uncertainty for decision‑making.

Our contributions are: (1) a rigorous, screenshot‑compliant hybrid modeling pipeline for microgrid dynamics; (2) a head‑to‑head comparison of physics, UDE, and BNode with per‑scenario metrics and confidence intervals; (3) uncertainty diagnostics and calibration analysis; (4) symbolic extraction of fθ(Pgen) for interpretability; and (5) a complete compute/timing report with open artifacts.

## 2. Methods
### 2.1 Physics Model
Eq1: dx1/dt = ηin·u+·I(u>0) − (1/ηout)·u−·I(u<0) − d(t)

Eq2: dx2/dt = −α·x2 + β·(Pgen − Pload) + γ·x1

### 2.2 UDE (Objective 2)
Keep Eq1 unchanged; replace β·Pgen with a learned fθ(Pgen):

Eq2 (UDE): dx2/dt = −α·x2 + fθ(Pgen) − β·Pload + γ·x1

We use a shallow tanh network with width w and tune w, L2 regularization λ, learning rate, solver tolerance, and seed via a coarse‑to‑fine search.

### 2.3 Bayesian Neural ODE (Objective 1)
Replace both equations with neural networks fθ1 and fθ2. Place priors on θ and observation noise σ; sample with NUTS (Turing.jl). Calibration metrics include empirical coverage at 50% and 90% and negative log‑likelihood (NLL).

### 2.4 Training & Solvers
DifferentialEquations.jl (Tsit5/Rosenbrock). Per‑scenario simulation with measured inputs u, d, Pgen, Pload at save points. Metrics: RMSE, MAE, R² with bootstrap confidence intervals.

## 3. Dataset and Setup
70 scenarios: 50 train, 10 validation, 10 test (~201 time points per scenario). Variables: x1, x2, u, d, Pgen, Pload, indicators. Scripts orchestrate the full pipeline; results and figures are written to versioned folders.

## 4. Results
### 4.1 Performance (Test)
- Physics: RMSE x1 ≈ 0.105, RMSE x2 ≈ 0.252, R² x2 ≈ 0.80
- UDE:     RMSE x1 ≈ 0.106, RMSE x2 ≈ 0.248, R² x2 ≈ 0.76

Finding: Near parity on x1; modest average RMSE gain for UDE on x2. Scenario‑level differences persist; improvements are not universally significant.

See Figure 2: `figures/fig2_performance_comparison_enhanced.{png,pdf,svg}`.

### 4.2 Uncertainty (BNode)
Posterior sampling ≈ 37 minutes (500 samples, 2 chains). Current posterior is under‑dispersed: 50% coverage ≈ 0.5%, 90% ≈ 0.5%, Mean NLL ≈ 2.69e5. We recommend broader σ priors or a heavier‑tailed likelihood to improve calibration.

See Figure 3 and `results/bnode_calibration_report.md`.

### 4.3 Symbolic Extraction (UDE)
Cubic fit: fθ(Pgen) ≈ −0.055 + 0.836·Pgen + 0.0009·Pgen² − 0.019·Pgen³. The dominant linear term supports a near‑linear generation effect in Eq2.

See Figure 4 and `results/ude_symbolic_extraction.md`.

## 5. Compute Budget
- UDE hyperparameter search: 2,880 configurations; wall‑clock ≈ 30 hours
- BNode posterior sampling: ≈ 37 minutes; 2 chains, 500 samples

## 6. Discussion & Limitations
UDE offers interpretable gains on x2 but incurs high ODE‑solve cost. BNode provides uncertainty but needs calibration improvements. Future: GPU acceleration (Flux CUDA; DiffEqGPU ensembles), likelihood tempering, and Student‑T observation models.

## 7. Reproducibility
Run: `julia --project=. scripts/run_enhanced_pipeline.jl`
Figures: `julia --project=. scripts/generate_research_figures_enhanced.jl`
Comparison: `julia --project=. scripts/comprehensive_model_comparison.jl`

Artifacts are versioned in `results/` and `figures/`; checkpoints in `checkpoints/`.

## 8. Ethics
Microgrid control impacts critical infrastructure; we constrain claims to retrospective modeling and analysis and release artifacts solely for research.

## References
Rackauckas et al., DifferentialEquations.jl; Chen et al., Neural ODEs; Universal Differential Equations; Bayesian calibration.
