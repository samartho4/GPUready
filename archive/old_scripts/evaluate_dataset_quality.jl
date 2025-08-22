#!/usr/bin/env julia

"""
    evaluate_dataset_quality.jl

Research-grade dataset diagnostics for roadmap objectives.
- Verifies presence of required variables (time, x1, x2, u, d, Pgen, Pload, scenario)
- Summarizes sizes, per-scenario counts, and time-grid properties
- Computes excitation/coverage metrics (charge/discharge balance, zero-crossings of Pgen-Pload)
- Correlations among exogenous signals and states
- Distribution summaries (mean/std/min/max/quantiles)
- Flags potential identifiability risks (high collinearity, low excitation)
- Writes a NeurIPS-ready markdown report to data/dataset_quality_report.md
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using Statistics, LinearAlgebra
using Dates

println("🔬 Evaluating dataset quality for roadmap objectives...")

# Candidate files in priority order
candidates = [
    ("train", joinpath(@__DIR__, "..", "data", "training_roadmap.csv")),
    ("val",   joinpath(@__DIR__, "..", "data", "validation_roadmap.csv")),
    ("test",  joinpath(@__DIR__, "..", "data", "test_roadmap.csv")),
]

# Include indicator/split control columns if present
required = [:time, :x1, :x2, :u, :d, :Pgen, :Pload, :scenario]

function must_have(df::DataFrame)
    all(sym -> sym in names(df), required)
end

function load_or_warn(label::String, path::String)
    if isfile(path)
        df = CSV.read(path, DataFrame)
        # Normalize to Symbol column names and strip whitespace
        orig = names(df)
        cleaned = map(n -> Symbol(strip(String(n))), orig)
        rename!(df, Pair.(orig, cleaned))
        # Derive indicator/split control columns if absent
        if :u in names(df)
            if !(:u_plus in names(df))
                df.u_plus = max.(0.0, df.u)
            end
            if !(:u_minus in names(df))
                df.u_minus = max.(0.0, -df.u)
            end
            if !(:I_u_pos in names(df))
                df.I_u_pos = Float64.(df.u .> 0)
            end
            if !(:I_u_neg in names(df))
                df.I_u_neg = Float64.(df.u .< 0)
            end
        end
        # Accept presence of derived indicator columns; only enforce base variables
        base_required = [:time, :x1, :x2, :u, :d, :Pgen, :Pload, :scenario]
        have = Set(Symbol.(names(df)))
        if !all(in(have), base_required)
            missing = [s for s in base_required if !(s in have)]
            error("$(label) file missing required columns: $(missing). Path=$(path)")
        end
        return df
    else
        error("$(label) file not found: $(path)")
    end
end

train_df = load_or_warn("train", candidates[1][2])
val_df   = load_or_warn("val",   candidates[2][2])
test_df  = load_or_warn("test",  candidates[3][2])

function scenario_groups(df::DataFrame)
    Dict(string(s) => sort(sub, :time) for sub in groupby(df, :scenario) for s in unique(string.(sub.scenario)))
end

train_sc = scenario_groups(train_df)
val_sc   = scenario_groups(val_df)
test_sc  = scenario_groups(test_df)

# Basic summaries
function basic_stats(df::DataFrame)
    (; nrows=nrow(df), nsc=length(unique(df.scenario)), tmin=minimum(df.time), tmax=maximum(df.time))
end

btrain = basic_stats(train_df)
bval   = basic_stats(val_df)
btest  = basic_stats(test_df)

# Excitation / coverage
zero_crossings(v) = sum(v[2:end] .* v[1:end-1] .< 0)

function scenario_diag(df::DataFrame)
    u = Vector{Float64}(df.u)
    d = Vector{Float64}(df.d)
    Pgen = Vector{Float64}(df.Pgen)
    Pload = Vector{Float64}(df.Pload)
    diff = Pgen .- Pload
    frac_charge = mean(u .> 0)
    zc = zero_crossings(diff)
    corr_gp = cor(Pgen, Pload)
    return (; frac_charge, zc, corr_gp,
        mean_u = mean(u), std_u = std(u),
        mean_d = mean(d), std_d = std(d),
        mean_diff = mean(diff), std_diff = std(diff))
end

function aggregate_diags(scdict::AbstractDict)
    diags = [scenario_diag(scdict[k]) for k in keys(scdict)]
    f = [d.frac_charge for d in diags]
    z = [d.zc for d in diags]
    r = [d.corr_gp for d in diags]
    return (
        mean_frac_charge = mean(f), std_frac_charge = std(f),
        mean_zero_cross = mean(z), min_zero_cross = minimum(z), max_zero_cross = maximum(z),
        mean_corr_gp = mean(r), max_abs_corr_gp = maximum(abs.(r))
    )
end

atrain = aggregate_diags(train_sc)
aval   = aggregate_diags(val_sc)
atest  = aggregate_diags(test_sc)

# Global correlations across splits
function global_corr(df::DataFrame)
    M = Matrix(df[:, [:u, :d, :Pgen, :Pload, :x1, :x2]])
    C = cor(M)
    vars = [:u, :d, :Pgen, :Pload, :x1, :x2]
    return (vars=vars, C=C)
end

Ctrain = global_corr(train_df)
Cval   = global_corr(val_df)
Ctest  = global_corr(test_df)

# Quantiles for key vars
function qtiles(df::DataFrame, syms::Vector{Symbol}; qs=(0.01, 0.05, 0.5, 0.95, 0.99))
    Dict(sym => [quantile(skipmissing(df[!, sym]), q) for q in qs] for sym in syms)
end

qvars = [:u, :d, :Pgen, :Pload, :x1, :x2]
qtrain = qtiles(train_df, qvars)
qval   = qtiles(val_df, qvars)
qtest  = qtiles(test_df, qvars)

# Flags
function flags(agg)
    return (
        warn_collinearity = agg.mean_corr_gp > 0.9 || agg.max_abs_corr_gp > 0.95,
        warn_low_excitation = agg.min_zero_cross < 3,
        target_balance = abs(agg.mean_frac_charge - 0.5) < 0.1
    )
end

ftrain = flags(atrain)
fval   = flags(aval)
ftest  = flags(atest)

# Write markdown report
out_dir = joinpath(@__DIR__, "..", "data")
if !isdir(out_dir); mkdir(out_dir); end
out_md = joinpath(out_dir, "dataset_quality_report.md")
open(out_md, "w") do io
    write(io, "# Dataset Quality Report (Roadmap Objectives)\n\n")
    write(io, "Date: $(now())\n\n")
    write(io, "## Variable Presence\n")
    write(io, "Required columns: $(required) — present in train/val/test ✔️\n\n")

    write(io, "## Sizes\n")
    write(io, "- Train: $(btrain.nrows) rows, $(btrain.nsc) scenarios, time ∈ [$(btrain.tmin), $(btrain.tmax)]\n")
    write(io, "- Val:   $(bval.nrows) rows, $(bval.nsc) scenarios, time ∈ [$(bval.tmin), $(bval.tmax)]\n")
    write(io, "- Test:  $(btest.nrows) rows, $(btest.nsc) scenarios, time ∈ [$(btest.tmin), $(btest.tmax)]\n\n")

    write(io, "## Excitation / Coverage Metrics\n")
    for (name, a, f) in [("Train", atrain, ftrain), ("Val", aval, fval), ("Test", atest, ftest)]
        write(io, "### $(name)\n")
        write(io, "- Mean charge fraction (u>0): $(round(a.mean_frac_charge, digits=3)) ± $(round(a.std_frac_charge, digits=3))\n")
        write(io, "- Zero-crossings of (Pgen−Pload) per scenario: mean=$(round(a.mean_zero_cross, digits=2)) [min=$(a.min_zero_cross), max=$(a.max_zero_cross)]\n")
        write(io, "- corr(Pgen,Pload): mean=$(round(a.mean_corr_gp, digits=3)); max |corr|=$(round(a.max_abs_corr_gp, digits=3))\n")
        write(io, "- Balance target (≈50% charging): $(f.target_balance ? "OK" : "偏")\n")
        write(io, "- Collinearity warning: $(f.warn_collinearity ? "YES" : "no")\n")
        write(io, "- Low-excitation warning (few crossings): $(f.warn_low_excitation ? "YES" : "no")\n\n")
    end

    write(io, "## Global Correlations (train)\n")
    write(io, "Variables: $(Ctrain.vars)\n\n")
    for i in 1:length(Ctrain.vars)
        row = [round(Ctrain.C[i,j], digits=3) for j in 1:length(Ctrain.vars)]
        write(io, "- $(Ctrain.vars[i]): $(row)\n")
    end
    write(io, "\n")

    write(io, "## Distribution Quantiles\n")
    function write_qblock(lbl, qd)
        write(io, "### $(lbl)\n")
        for sym in qvars
            qs = qd[sym]
            write(io, "- $(sym): q01=$(round(qs[1], digits=3)), q05=$(round(qs[2], digits=3)), q50=$(round(qs[3], digits=3)), q95=$(round(qs[4], digits=3)), q99=$(round(qs[5], digits=3))\n")
        end
        write(io, "\n")
    end
    write_qblock("Train", qtrain)
    write_qblock("Val", qval)
    write_qblock("Test", qtest)

    write(io, "## Readiness for Objectives\n")
    write(io, "- Objective 2 compliance: variables u, d, Pgen, Pload present → Eq1 physics-only and Eq2 fθ(Pgen,Pload) are implementable.\n")
    write(io, "- Identifiability: independent variation encouraged via DOE segments; check collinearity/zero-crossings flags above.\n")
    write(io, "- Per-scenario structure: evaluation should initialize each scenario from its first observed state (recommended).\n\n")

    write(io, "## Risks & Mitigations\n")
    write(io, "- High corr(Pgen,Pload) in some scenarios → mitigate with additional DOE segments or random phase differences.\n")
    write(io, "- Few zero-crossings in some clips → extend Tfinal or increase segment count.\n")
    write(io, "- Extreme clipping at x1 bounds → adjust control/demand ranges to reduce saturation.\n")
end

println("📝 Wrote dataset quality report → data/dataset_quality_report.md") 