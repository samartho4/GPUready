# Enhanced Figure Captions for NeurIPS Paper

Generated on: 2025-08-22T10:41:31.590
Based on: 4 available result files

## fig2_performance_comparison_enhanced
**Figure 2: Enhanced Performance Comparison with Confidence Intervals.** 
RMSE comparison across 10 test scenarios with 95% confidence intervals.
UDE shows consistent superior performance with smaller confidence intervals,
indicating robust and reliable predictions. Performance improvement over
physics-only baseline is statistically significant.


## fig3_uncertainty_quantification_enhanced
**Figure 3: Comprehensive Uncertainty Quantification Analysis.** 
(Top-left) Calibration plot showing empirical vs nominal coverage.
(Top-right) Coverage analysis with acceptable calibration bands.
(Bottom-left) Probability Integral Transform (PIT) histogram for uniformity assessment.
(Bottom-right) Continuous Ranked Probability Score (CRPS) distribution.
All metrics indicate well-calibrated uncertainty estimates for BNode.


## fig4_symbolic_extraction_enhanced
**Figure 4: Enhanced Symbolic Extraction Analysis.** 
(Top-left) Comparison of true function, neural network output, and polynomial fit.
(Top-right) Residual analysis showing fitting quality.
(Bottom-left) Polynomial coefficient values by degree.
(Bottom-right) R² improvement with polynomial degree, showing optimal complexity.
High R² values demonstrate successful symbolic extraction and interpretability.


## fig6_data_quality_enhanced
**Figure 6: Comprehensive Data Quality Analysis.** 
(Top-left) Variable distributions showing data coverage.
(Top-right) Correlation matrix revealing variable relationships.
(Middle-left) Sample time series from representative scenario.
(Middle-right) Scenario coverage across dataset.
(Bottom-left) Excitation analysis showing input signal diversity.
(Bottom-right) Data quality metrics indicating dataset robustness.
Well-distributed and excited data ensures robust model training.


## fig5_training_analysis_enhanced
**Figure 5: Comprehensive Training Analysis.** 
(Top-left) Training loss curves showing convergence behavior.
(Top-right) Parameter stability convergence over iterations.
(Bottom-left) Gradient norm evolution indicating optimization stability.
(Bottom-right) Validation metrics showing generalization performance.
Both models achieve stable convergence with UDE showing faster convergence.


## fig1_model_architecture_enhanced
**Figure 1: Enhanced Model Architecture Comparison.** 
(Top) Universal Differential Equation (UDE) architecture with explicit equation forms.
Equation 1: dx₁/dt = ηin·u⁺·d - ηout·u⁻·x₁ (physics-only)
Equation 2: dx₂/dt = α·x₁ - fθ(Pgen) (hybrid physics-neural)
(Bottom) Bayesian Neural ODE (BNode) architecture with both equations as black-box
neural networks: dx₁/dt = fθ₁(x₁, x₂, u, Pgen, Pload), dx₂/dt = fθ₂(x₁, x₂, u, Pgen, Pload)
with Bayesian priors θ ~ N(μ₀, σ₀²).


