# Dataset Quality Report (Roadmap Objectives)

Date: 2025-08-18T17:01:36.499

## Variable Presence
Required columns: [:time, :x1, :x2, :u, :d, :Pgen, :Pload, :scenario] — present in train/val/test ✔️

## Sizes
- Train: 10050 rows, 50 scenarios, time ∈ [0.0, 10.0]
- Val:   2010 rows, 10 scenarios, time ∈ [0.0, 10.0]
- Test:  2010 rows, 10 scenarios, time ∈ [0.0, 10.0]

## Excitation / Coverage Metrics
### Train
- Mean charge fraction (u>0): 0.491 ± 0.058
- Zero-crossings of (Pgen−Pload) per scenario: mean=16.6 [min=0, max=56]
- corr(Pgen,Pload): mean=0.056; max |corr|=0.698
- Balance target (≈50% charging): OK
- Collinearity warning: no
- Low-excitation warning (few crossings): YES

### Val
- Mean charge fraction (u>0): 0.542 ± 0.062
- Zero-crossings of (Pgen−Pload) per scenario: mean=12.2 [min=0, max=35]
- corr(Pgen,Pload): mean=-0.18; max |corr|=0.659
- Balance target (≈50% charging): OK
- Collinearity warning: no
- Low-excitation warning (few crossings): YES

### Test
- Mean charge fraction (u>0): 0.512 ± 0.058
- Zero-crossings of (Pgen−Pload) per scenario: mean=18.2 [min=1, max=50]
- corr(Pgen,Pload): mean=-0.064; max |corr|=0.723
- Balance target (≈50% charging): OK
- Collinearity warning: no
- Low-excitation warning (few crossings): YES

## Global Correlations (train)
Variables: [:u, :d, :Pgen, :Pload, :x1, :x2]

- u: [1.0, -0.001, -0.039, -0.052, 0.006, -0.091]
- d: [-0.001, 1.0, 0.1, 0.113, -0.241, -0.048]
- Pgen: [-0.039, 0.1, 1.0, 0.081, 0.015, 0.376]
- Pload: [-0.052, 0.113, 0.081, 1.0, -0.075, -0.292]
- x1: [0.006, -0.241, 0.015, -0.075, 1.0, 0.137]
- x2: [-0.091, -0.048, 0.376, -0.292, 0.137, 1.0]

## Distribution Quantiles
### Train
- u: q01=-1.0, q05=-1.0, q50=-0.012, q95=1.0, q99=1.0
- d: q01=0.025, q05=0.064, q50=0.226, q95=0.424, q99=0.47
- Pgen: q01=0.1, q05=0.1, q50=0.631, q95=1.179, q99=1.271
- Pload: q01=0.1, q05=0.1, q50=0.579, q95=1.193, q99=1.278
- x1: q01=-3.866, q05=-3.219, q50=-0.647, q95=1.424, q99=2.362
- x2: q01=-3.19, q05=-1.706, q50=0.063, q95=2.065, q99=2.925

### Val
- u: q01=-1.0, q05=-1.0, q50=0.05, q95=1.0, q99=1.0
- d: q01=0.021, q05=0.067, q50=0.25, q95=0.436, q99=0.476
- Pgen: q01=0.1, q05=0.229, q50=0.712, q95=1.205, q99=1.288
- Pload: q01=0.1, q05=0.1, q50=0.657, q95=1.182, q99=1.272
- x1: q01=-4.127, q05=-3.683, q50=-0.622, q95=1.325, q99=1.968
- x2: q01=-2.874, q05=-2.094, q50=-0.06, q95=1.452, q99=2.178

### Test
- u: q01=-1.0, q05=-1.0, q50=0.019, q95=1.0, q99=1.0
- d: q01=0.049, q05=0.097, q50=0.321, q95=0.45, q99=0.492
- Pgen: q01=0.1, q05=0.1, q50=0.607, q95=1.175, q99=1.275
- Pload: q01=0.1, q05=0.1, q50=0.597, q95=1.183, q99=1.279
- x1: q01=-4.894, q05=-4.179, q50=-0.583, q95=1.257, q99=1.895
- x2: q01=-2.832, q05=-1.869, q50=0.086, q95=1.754, q99=2.696

## Readiness for Objectives
- Objective 2 compliance: variables u, d, Pgen, Pload present → Eq1 physics-only and Eq2 fθ(Pgen,Pload) are implementable.
- Identifiability: independent variation encouraged via DOE segments; check collinearity/zero-crossings flags above.
- Per-scenario structure: evaluation should initialize each scenario from its first observed state (recommended).

## Risks & Mitigations
- High corr(Pgen,Pload) in some scenarios → mitigate with additional DOE segments or random phase differences.
- Few zero-crossings in some clips → extend Tfinal or increase segment count.
- Extreme clipping at x1 bounds → adjust control/demand ranges to reduce saturation.
