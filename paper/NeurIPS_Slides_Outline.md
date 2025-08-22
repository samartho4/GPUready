# NeurIPS Slides Outline

1. Title & Authors
   - Title: Stability Without Discovery — Why Validation Beyond R² Matters
   - Authors, affiliations

2. Motivation & Problem
   - Short microgrid schematic (optional)
   - Goal: learn hidden physics + uncertainty

3. Prior Pitfalls
   - Bullet: misleading success; dead NN
   - Link (footnote): prior deck

4. Contributions
   - Reproducible pipeline; consistent metrics; diagnostics vs validation; OOD hooks

5. Data & Splits
   - Temporal split; scenario‑disjoint option; OOD flag
   - Code snippet (tiny) to reproduce

6. Models & Inference
   - BNN‑ODE (10 params); UDE = physics + NN residual
   - Strict tolerances; PPC/PIT

7. Figure 1 — Performance (Trajectory MSE)
   - Insert: `paper/figures/fig1_performance_comparison.png`
   - Caption: Values from table (0.16, 32.16, 16.45)

8. Diagnostic: Derivative MSE (text only)
   - Note: derivative MSE ~ 590.9/603.6/20.0 (not used in Fig 1)

9. Figure 2 — Discovery Diagnostic
   - Insert: `paper/figures/fig2_physics_discovery.png`
   - Caption: NN vs β(Pgen−Pload), no claim

10. Figure 3 — Symbolic Surrogate R²
   - Insert: `paper/figures/fig3_ude_symbolic_success.png`
   - Caption: R² vs NN output

11. Validation Gate
   - Insert: `paper/figures/fig_validation_gate.png`
   - Caption: |learned|≈0; |target|=1.2 → not validated

12. Uncertainty & Calibration
   - Insert: `paper/figures/ppc_bayesian_ode.png`, `ppc_ude.png`, `pit_bnn_x1.png`

13. Results Table (excerpt)
   - Paste small portion of `paper/results/final_results_table.md`

14. Limitations
   - Identifiability; residual collapse

15. Roadmap
   - Data excitation; constraints; SBC; PSIS‑LOO/WAIC

16. Reproducibility
   - One command `./bin/reproduce.sh`
   - Env flags for scenario‑disjoint/OOD

17. Backup & Q&A
   - Diagnostic vs metric; failure mode taxonomy 