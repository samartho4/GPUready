### Title
Microgrid Bayesian Neural/Universal Differential Equations: Stability Without Discovery — Why Validation Beyond R² Matters

Time: 12–15 min
Presenter: <your name>

Links to prior context: [BNODE for Microgrid — Lessons in Numerical Stability and Model Verification](https://bnode-for-microgrid-nz48frl.gamma.site)

---

### Slide 1 — Problem & Setting (1 min)
- Microgrid state estimation/control with physics + learning
- Goal: learn hidden physics term; quantify uncertainty; remain numerically stable
- Models: Bayesian Neural ODE (BNN‑ODE), Universal Differential Equation (UDE)

Speaker notes:
- We target interpretable, uncertainty‑aware dynamics. The hidden term is β·(Pgen−Pload).

---

### Slide 2 — What went wrong in early work (1 min)
- “Misleading success” and a “dead NN” (flat outputs)
- Stability ≠ learning (R² high vs NN output but physics terms ~0)
- Ref: prior deck summary: numerical stability alone is not enough [(link)](https://bnode-for-microgrid-nz48frl.gamma.site)

Speaker notes:
- Great looking trajectories and R² can coexist with zero physics content.

---

### Slide 3 — Our contributions in this submission (1 min)
- Reproducible end‑to‑end pipeline (`./bin/reproduce.sh`)
- Consistent metrics: Fig 1 uses trajectory MSE from results table
- Diagnostic separation: discovery diagnostic vs validation gate
- Scenario‑disjoint/OOD capability; honest negative result

---

### Slide 4 — Data & Splits (0.5 min)
- Temporal split: ~0–60 train, 60+ test
- Scenario leakage detector + optional scenario‑disjoint split (env flags)
- OOD hooks via `MG_OOD_SCENARIOS`

Demo note:
```bash
MG_SPLIT=scenario MG_TRAIN_SCENARIOS=S1-1,S1-2,S1-3 MG_TEST_SCENARIOS=S1-4,S1-5 ./bin/reproduce.sh
```

---

### Slide 5 — Models & Inference (0.5 min)
- BNN‑ODE baseline (10 params), UDE = 5 physics + NN residual
- Strict ODE tolerances (abstol=reltol=1e‑8)
- Posterior predictive checks (PPC), PIT

---

### Slide 6 — Fig 1: Performance (1 min)
- Insert: `paper/figures/fig1_performance_comparison.png`
- Values from table: Physics‑only 0.16; BNN‑ODE 32.16; UDE 16.45
- Log‑scale axis; title states “Trajectory Simulation MSE”

Demo cue (optional):
```bash
julia --project=. -e 'include("scripts/generate_figures.jl")'
# Look for: Figure 1 trajectory MSE values (Physics-only, BNN-ODE, UDE): [0.16, 32.16, 16.45]
```

---

### Slide 7 — Diagnostics: Derivative MSE (0.5 min)
- The dynamic derivative MSE prints (≈590.9, 603.6, 20.0) are diagnostics only
- Not used in Fig 1; helps sanity‑check gradients vs data differences

---

### Slide 8 — Fig 2: Discovery Diagnostic (1 min)
- Insert: `paper/figures/fig2_physics_discovery.png`
- NN output vs β(Pgen−Pload); we do NOT claim discovery here

---

### Slide 9 — Fig 3: Symbolic Surrogate R² (1 min)
- Insert: `paper/figures/fig3_ude_symbolic_success.png`
- R² vs NN output; helps surrogatize residual; not a physics claim

---

### Slide 10 — Validation Gate (1 min)
- Insert: `paper/figures/fig_validation_gate.png`
- Learned |Pgen|,|Pload| ≪ |β|; physics discovery not validated
- Key insight: high R² on NN residual ≠ recovered physics

---

### Slide 11 — Uncertainty & Calibration (1 min)
- Insert: `paper/figures/ppc_bayesian_ode.png`, `ppc_ude.png`, `pit_bnn_x1.png`
- Coverage and PIT are miscalibrated → signals misspecification / dead residual

---

### Slide 12 — Negative Result as Contribution (1 min)
- Numerically stable training + verification is necessary but not sufficient
- Honest failure case; checklist: trajectory MSE, discovery diagnostic, validation gate, PPC/PIT
- Aligns with guidance in prior reflection [(link)](https://bnode-for-microgrid-nz48frl.gamma.site)

---

### Slide 13 — What changed since prior deck (0.5 min)
- Fig 1 now consistent with table; pipeline prints used values
- Added validation gate figure + explicit failure
- Scenario‑disjoint/OOD hooks and README reviewer guide

---

### Slide 14 — Limitations (0.5 min)
- Residual collapses toward zero; identifiability issues
- Symbolic extraction matches NN, not physics

---

### Slide 15 — Roadmap (1 min)
- Data: OOD, interventions, richer excitation
- Priors/constraints: monotonicity/ICNN, guidance priors on β
- Inference: non‑centered params, mass matrix adaptation, SBC
- Model selection: PSIS‑LOO/WAIC with ELPD deltas

---

### Live Demo (optional, +2–3 min)
```bash
./bin/reproduce.sh
# Look for:
#  - Figure 1 trajectory MSE values (Physics-only, BNN-ODE, UDE): [0.16, 32.16, 16.45]
#  - Validation gate values (|learned|, |target|): [0.0003, 0.0005], [1.2, 1.2]
```
Open figures from `paper/figures/` as you narrate.

---

### Backup / Q&A
- Why derivative MSE so large vs trajectory MSE? Different target; diagnostic only
- Why physics not discovered? Dead residual + weak identifiability; need constraints/prior structure + excitation
- Are results reproducible? Yes: one command + printed values in logs + mirrored artifacts

---

### Appendix — Reproduction Details
- Command: `./bin/reproduce.sh`
- Scenario‑disjoint/OOD: `MG_SPLIT=scenario MG_TRAIN_SCENARIOS=... MG_TEST_SCENARIOS=... MG_OOD_SCENARIOS=...`
- Artifacts: `paper/figures/*.png`, `paper/results/final_results_table.md`, `paper/results/table1_symbolic_results.txt` 