# Results Interpretation
 
- Reading tables with CI and significance
- Understanding physics validation metrics
- Calibration and uncertainty plots 

## Summary
- **UDE tuning completed successfully** after ~30 hours (much longer than estimated 4.8 hours due to ODE solving complexity).
- **BNode training completed** in 37 minutes with MCMC sampling; posterior calibrated metrics currently under-cover (50%≈0.5%, 90%≈0.5%).
- **Comprehensive comparison** on test set shows physics and UDE are comparable on x1; UDE slightly improves x2 RMSE vs physics.
- **Symbolic extraction** yields a cubic for fθ(Pgen) with dominant linear term (~0.836).

## Key numbers
- **UDE tuning**: 2,880 configurations tested over ~30 hours (vs estimated 4.8 hours)
- **BNode training**: 37 minutes for MCMC sampling (500 samples, 2 chains)
- **Test (physics)**: RMSE x1 ≈ 0.105, RMSE x2 ≈ 0.252, R² x2 ≈ 0.80
- **Test (UDE)**: RMSE x1 ≈ 0.106, RMSE x2 ≈ 0.248, R² x2 ≈ 0.76
- **BNode calibration**: 50% coverage ≈ 0.005, 90% coverage ≈ 0.005; Mean NLL ≈ 2.69e5

## Interpretation
- **Physics vs UDE**: Near parity on storage state (x1). UDE offers slightly better power-flow (x2) RMSE but slightly lower R² than physics; differences are small.
- **BNode uncertainty**: Severe under-coverage indicates posterior scale is too tight; increase observation noise prior scale or use likelihood tempering.
- **Symbolic fθ(Pgen)**: Linear component dominates; higher-order terms are small, supporting a near-linear effect of generation on dynamics.
- **Timing insights**: UDE tuning took ~30 hours vs estimated 4.8 hours, highlighting ODE solving complexity. BNode training was efficient at 37 minutes.

## Next improvements
- **Retune BNode likelihood/prior** (e.g., broader σ prior or StudentT likelihood) to improve coverage.
- **Explore modest UDE width increase** and regularization sweep around the best region.
- **Consider ensembling or bootstrap** to quantify variability in test metrics.
- **Optimize UDE tuning** for faster convergence (e.g., early stopping, adaptive search). 