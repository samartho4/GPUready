# Roadmap Dataset Generation Summary

Date: 2025-08-18T16:49:40.991

## Sizes
- Train rows: 10050 in 50 scenarios
- Val rows:   2010 in 10 scenarios
- Test rows:  2010 in 10 scenarios

## Train diagnostics
- Mean charge fraction (u>0): 0.491 ± 0.058
- Zero-crossings of (Pgen−Pload) per scenario: mean=16.6 [min=0, max=56]
- Mean corr(Pgen,Pload) = 0.056; max |corr| = 0.698

## Val diagnostics
- Mean charge fraction (u>0): 0.542 ± 0.062
- Zero-crossings of (Pgen−Pload) per scenario: mean=12.2 [min=0, max=35]
- Mean corr(Pgen,Pload) = -0.18; max |corr| = 0.659

## Test diagnostics
- Mean charge fraction (u>0): 0.512 ± 0.058
- Zero-crossings of (Pgen−Pload) per scenario: mean=18.2 [min=1, max=50]
- Mean corr(Pgen,Pload) = -0.064; max |corr| = 0.723

## Flags
- High Pgen/Pload collinearity present? No
- Low excitation (few zero-crossings) present? Yes
