#!/usr/bin/env julia

"""
    generate_roadmap_dataset.jl

Generate synthetic microgrid dataset fully aligned with the roadmap equations:
  Eq1: dx1/dt = ηin * u+(t) - (1/ηout) * u−(t) - d(t)
  Eq2: dx2/dt = -α x2 + β * (Pgen(t) - Pload(t)) + γ * x1

Improvements for research quality:
  - Scenario-level physics sampling (ηin, ηout, α, γ, β)
  - DOE-style exogenous signals with independent variation of Pgen and Pload
  - Balanced charge/discharge and sign changes of (Pgen-Pload)
  - Diagnostics: coverage stats, zero-crossings, correlations, hist summaries

Outputs:
  data/training_roadmap.csv
  data/validation_roadmap.csv
  data/test_roadmap.csv
  data/roadmap_generation_summary.md
"""

using Pkg
Pkg.activate(".")

using DifferentialEquations
using Random, Statistics, LinearAlgebra
using CSV, DataFrames
using Dates

println("🧪 Generating roadmap-aligned dataset (>7k points)...")

# Global random seed for reproducibility
Random.seed!(42)

# Scenario counts and temporal grid
num_train = 50
num_val   = 10
num_test  = 10
Tfinal    = 10.0
Δt        = 0.05
saveat    = collect(0.0:Δt:Tfinal)  # 201 points per scenario

# Physics priors (for data generation only)
ηin_range  = (0.88, 0.98)
ηout_range = (0.85, 0.95)
α_range    = (0.05, 0.20)
γ_range    = (0.01, 0.05)
β_range    = (0.8, 1.2)

# Utility
unif(a,b) = a + (b-a) * rand()
clmp01(x) = clamp(x, 0.0, 1.0)

# Build DOE-style exogenous signals
function make_exogenous(T::Vector{Float64}; seed::Int, target_charge_frac::Float64=0.5)
    rng = MersenneTwister(seed)
    n = length(T)
    # 1) Smooth carriers (sinusoids) to avoid overly synthetic steps only
    φ1, φ2, φ3 = 2π*rand(rng, 3)
    base1 = 0.6 .+ 0.35 .* sin.(0.25 .* T .+ φ1)
    base2 = 0.6 .+ 0.35 .* sin.(0.33 .* T .+ φ2)
    noise = 0.05 .* randn(rng, n)
    # 2) DOE segments: piecewise ramps/steps where we vary Pgen and Pload independently
    seg_len = max(10, Int(floor(n ÷ 8)))
    Pgen = copy(base1)
    Pload = copy(base2)
    for k in 1:8
        i0 = (k-1)*seg_len + 1
        i1 = min(k*seg_len, n)
        if isodd(k)
            # vary Pgen, keep Pload near baseline
            Pgen[i0:i1] .+= 0.3 .* sign(rand(rng) - 0.5)
        else
            # vary Pload, keep Pgen near baseline
            Pload[i0:i1] .+= 0.3 .* sign(rand(rng) - 0.5)
        end
    end
    Pgen .+= noise; Pload .-= noise
    Pgen .= clamp.(Pgen, 0.1, 1.6)
    Pload .= clamp.(Pload, 0.1, 1.6)

    # 3) Control u(t): balanced charge/discharge with random phases; then bias-correct
    u_raw = 0.8 .* sin.(0.7 .* T .+ φ3) .+ 0.6 .* sin.(1.1 .* T .+ 2π*rand(rng)) .+ 0.1 .* randn(rng, n)
    # Bias to reach target charge fraction
    frac_pos = mean(u_raw .> 0)
    bias = clamp(target_charge_frac - frac_pos, -0.2, 0.2)
    u = clamp.(u_raw .+ bias, -1.0, 1.0)

    # 4) Demand d(t): positive, slowly varying
    d = abs.(0.25 .+ 0.15 .* sin.(0.2 .* T .+ 2π*rand(rng)) .+ 0.05 .* randn(rng, n))

    return (u, d, Pgen, Pload)
end

# RHS with scenario-specific physics
function rhs!(du, x, p, t, u_sig, d_sig, Pgen_sig, Pload_sig, Tvec, ηin, ηout, α, γ, β)
    x1, x2 = x
    idx = clamp(searchsortedlast(Tvec, t), 1, length(Tvec))
    u_t, d_t = u_sig[idx], d_sig[idx]
    Pgen_t, Pload_t = Pgen_sig[idx], Pload_sig[idx]
    up, un = max(0.0, u_t), max(0.0, -u_t)
    du[1] = ηin * up - (1/ηout) * un - d_t
    du[2] = -α * x2 + β * (Pgen_t - Pload_t) + γ * x1
end

# Diagnostics per scenario
struct ScenarioDiag
    scenario::String
    frac_charge::Float64
    frac_discharge::Float64
    mean_gen_minus_load::Float64
    zero_crossings::Int
    corr_Pgen_Pload::Float64
end

function compute_diag(sid::String, T, u, d, Pgen, Pload)
    frac_charge = mean(u .> 0)
    frac_discharge = 1 - frac_charge
    diff = Pgen .- Pload
    zero_cross = sum(diff[2:end] .* diff[1:end-1] .< 0)
    C = cor(hcat(Pgen, Pload))
    corr_gp = C[1,2]
    return ScenarioDiag(sid, frac_charge, frac_discharge, mean(diff), zero_cross, corr_gp)
end

function simulate_one(sid::String; seed::Int)
    T = saveat
    # Sample physics per scenario
    ηin  = unif(ηin_range...)
    ηout = unif(ηout_range...)
    α    = unif(α_range...)
    γ    = unif(γ_range...)
    β    = unif(β_range...)

    u_sig, d_sig, Pgen_sig, Pload_sig = make_exogenous(T; seed)
    # Initial state
    x10 = clmp01(0.5 + 0.2*randn())
    x20 = 0.1 * randn()

    prob = ODEProblem((du,u,p,t)->rhs!(du,u,p,t,u_sig,d_sig,Pgen_sig,Pload_sig,T,ηin,ηout,α,γ,β), [x10, x20], (0.0, T[end]))
    sol = solve(prob, Tsit5(); saveat=T, abstol=1e-6, reltol=1e-6)
    if sol.retcode != :Success
        error("Scenario $(sid) failed to integrate")
    end
    X = permutedims(reduce(hcat, (sol(t) for t in T)))

    # Indicator-based representation to mirror Eq1 exactly
    Ipos = float.(u_sig .> 0)
    Ineg = float.(u_sig .< 0)
    up   = max.(0.0, u_sig)
    un   = max.(0.0, -u_sig)

    df = DataFrame(time=T, x1=X[:,1], x2=X[:,2], u=u_sig, u_plus=up, u_minus=un,
                   I_u_pos=Ipos, I_u_neg=Ineg, d=d_sig, Pgen=Pgen_sig, Pload=Pload_sig)
    df.scenario .= sid
    df.ηin .= ηin; df.ηout .= ηout; df.α .= α; df.γ .= γ; df.β .= β

    diag = compute_diag(sid, T, u_sig, d_sig, Pgen_sig, Pload_sig)
    return df, diag
end

function make_split(nsc::Int, prefix::String, start_seed::Int)
    dfs = DataFrame[]
    diags = ScenarioDiag[]
    for i in 1:nsc
        sid = "$(prefix)-$(i)"
        df, d = simulate_one(sid; seed=start_seed + i)
        push!(dfs, df)
        push!(diags, d)
    end
    return vcat(dfs...), diags
end

train_df, train_diags = make_split(num_train, "train", 1000)
val_df,   val_diags   = make_split(num_val,   "val",   2000)
test_df,  test_diags  = make_split(num_test,  "test",  3000)

mkpath("data")
CSV.write(joinpath("data","training_roadmap.csv"), train_df)
CSV.write(joinpath("data","validation_roadmap.csv"), val_df)
CSV.write(joinpath("data","test_roadmap.csv"), test_df)

# -----------------------------------------------------------------------------
# Diagnostics summary
# -----------------------------------------------------------------------------
function summarize(diags::Vector{ScenarioDiag})
    frac_charge = [d.frac_charge for d in diags]
    zc = [d.zero_crossings for d in diags]
    corr_gp = [d.corr_Pgen_Pload for d in diags]
    return (
        mean_frac_charge = mean(frac_charge),
        std_frac_charge = std(frac_charge),
        mean_zero_crossings = mean(zc),
        min_zero_crossings = minimum(zc),
        max_zero_crossings = maximum(zc),
        mean_corr_gp = mean(corr_gp),
        max_abs_corr_gp = maximum(abs.(corr_gp))
    )
end

sum_train = summarize(train_diags)
sum_val   = summarize(val_diags)
sum_test  = summarize(test_diags)

report_path = joinpath("data","roadmap_generation_summary.md")
open(report_path, "w") do io
    write(io, "# Roadmap Dataset Generation Summary\n\n")
    write(io, "Date: $(now())\n\n")
    write(io, "## Sizes\n")
    write(io, "- Train rows: $(nrow(train_df)) in $(length(train_diags)) scenarios\n")
    write(io, "- Val rows:   $(nrow(val_df)) in $(length(val_diags)) scenarios\n")
    write(io, "- Test rows:  $(nrow(test_df)) in $(length(test_diags)) scenarios\n\n")

    for (name, s) in [("Train", sum_train), ("Val", sum_val), ("Test", sum_test)]
        write(io, "## $(name) diagnostics\n")
        write(io, "- Mean charge fraction (u>0): $(round(s.mean_frac_charge, digits=3)) ± $(round(s.std_frac_charge, digits=3))\n")
        write(io, "- Zero-crossings of (Pgen−Pload) per scenario: mean=$(round(s.mean_zero_crossings, digits=2)) [min=$(s.min_zero_crossings), max=$(s.max_zero_crossings)]\n")
        write(io, "- Mean corr(Pgen,Pload) = $(round(s.mean_corr_gp, digits=3)); max |corr| = $(round(s.max_abs_corr_gp, digits=3))\n\n")
    end

    # Simple flags
    warn_corr = any(abs(d.corr_Pgen_Pload) > 0.9 for d in vcat(train_diags, val_diags, test_diags))
    warn_zc = any(d.zero_crossings < 3 for d in vcat(train_diags, val_diags, test_diags))
    write(io, "## Flags\n")
    write(io, "- High Pgen/Pload collinearity present? $(warn_corr ? "Yes" : "No")\n")
    write(io, "- Low excitation (few zero-crossings) present? $(warn_zc ? "Yes" : "No")\n")
end

println("✅ Generated:")
println("  data/training_roadmap.csv  → $(nrow(train_df)) rows")
println("  data/validation_roadmap.csv → $(nrow(val_df)) rows")
println("  data/test_roadmap.csv      → $(nrow(test_df)) rows")
println("📄 Diagnostics: data/roadmap_generation_summary.md") 