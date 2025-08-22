# Figure Captions for NeurIPS Paper

Generated on: 2025-08-20T09:25:31.964

## fig1_model_architecture
**Figure 1: Model Architecture Comparison.** 
(Top) Universal Differential Equation (UDE) architecture showing hybrid physics-neural approach. 
Equation 1 remains physics-only while Equation 2 replaces only β⋅Pgen(t) with neural network fθ(Pgen(t)). 
(Bottom) Bayesian Neural ODE (BNode) architecture with both equations as black-box neural networks 
and Bayesian priors on parameters.


## fig5_training_stability
**Figure 5: Training Stability Analysis.** 
Training loss curves for UDE and BNode models showing convergence behavior. 
Both models achieve stable convergence, with UDE showing slightly faster convergence 
due to physics-informed initialization and constraints.


## fig4_symbolic_extraction
**Figure 4: Symbolic Extraction Results.** 
Comparison of true function f(Pgen), learned neural network fθ(Pgen), and extracted polynomial fit. 
High R² value demonstrates successful symbolic extraction, enabling interpretability 
of the learned neural correction term.


## fig2_performance_comparison
**Figure 2: Performance Comparison Across Scenarios.** 
RMSE comparison of Physics-only baseline, UDE, and BNode models across 10 test scenarios. 
UDE shows superior performance by combining physics constraints with learned dynamics, 
while BNode provides uncertainty quantification at computational cost.


## fig6_data_quality
**Figure 6: Data Quality and Distribution.** 
Histograms of key variables (x1, x2, u, Pgen, Pload, d) showing data distribution 
across 10,050 training points from 50 scenarios. Well-distributed data ensures 
robust model training and generalization.


## fig3_uncertainty_calibration
**Figure 3: BNode Uncertainty Calibration.** 
Calibration plot showing empirical vs nominal coverage for BNode predictions. 
Points close to the diagonal line indicate well-calibrated uncertainty estimates. 
The gray band represents acceptable calibration range (±5%).


