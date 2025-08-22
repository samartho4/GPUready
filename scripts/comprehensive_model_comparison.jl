#!/usr/bin/env julia

"""
    comprehensive_model_comparison.jl

Compare Physics-only, UDE, and BNode on the roadmap test set with per-scenario simulation.
Also performs symbolic extraction of the UDE's fθ(Pgen) via polynomial fitting.

Outputs:
  - results/comprehensive_metrics.csv
  - results/comprehensive_comparison_summary.md
  - results/ude_symbolic_extraction.md
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using DifferentialEquations
using Statistics, Random
using BSON
using LinearAlgebra

# ------------------------------------------------------------
# Data loading and grouping
# ------------------------------------------------------------

test_csv = joinpath(@__DIR__, "..", "data", "test_roadmap.csv")
@assert isfile(test_csv) "Missing test_roadmap.csv"
test_df = CSV.read(test_csv, DataFrame)

function group_scenarios(df::DataFrame)
    Dict(string(s)=>sort(sub, :time) for sub in groupby(df, :scenario) for s in unique(string.(sub.scenario)))
end

test_sc = group_scenarios(test_df)

# Output directory
res_dir = joinpath(@__DIR__, "..", "results")
if !isdir(res_dir); mkdir(res_dir); end

# ------------------------------------------------------------
# UDE loading and functions (Objective 2)
# ------------------------------------------------------------

ckpt_dir = joinpath(@__DIR__, "..", "checkpoints")
ude_ckpt = joinpath(ckpt_dir, "ude_best_tuned.bson")
if !isfile(ude_ckpt)
    ude_ckpt = joinpath(ckpt_dir, "ude_roadmap_opt.bson")
end
@assert isfile(ude_ckpt) "No UDE checkpoint found"

ckpt_data = BSON.load(ude_ckpt)

ude_params = Vector{Float64}()
ude_width = 5
if haskey(ckpt_data, :best_ckpt)
    # Support both flat vector checkpoints and structured ones
    if ckpt_data[:best_ckpt] isa Vector{Float64}
        ude_params = ckpt_data[:best_ckpt]
        ude_width = haskey(ckpt_data, :best_cfg) ? ckpt_data[:best_cfg][1] : 5
    else
        ude_params = ckpt_data[:best_ckpt][:p]
        ude_width = ckpt_data[:best_ckpt][:width]
    end
elseif haskey(ckpt_data, :ude_params_opt)
    ude_params = ckpt_data[:ude_params_opt]
    ude_width = 5
else
    error("UDE checkpoint missing parameters")
end

function ftheta_ude(Pgen::Float64, θ::Vector{Float64}, width::Int)
    W1 = reshape(θ[1:width], width, 1)
    b1 = θ[width+1:width+width]
    h = tanh.(W1 * [Pgen] .+ b1)
    return sum(h)
end

function ude_rhs(params::Vector{Float64}, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        up_t, um_t = sdf.u_plus[idx], sdf.u_minus[idx]
        Ipos_t, Ineg_t = sdf.I_u_pos[idx], sdf.I_u_neg[idx]
        d_t = sdf.d[idx]
        Pgen_t, Pload_t = sdf.Pgen[idx], sdf.Pload[idx]
        ηin, ηout, α, β, γ = params[1:5]
        θ = params[6:end]
        du[1] = ηin * up_t * Ipos_t - (1/ηout) * um_t * Ineg_t - d_t
        du[2] = -α * x2 + ftheta_ude(Pgen_t, θ, width) - β * Pload_t + γ * x1
    end
    return rhs!
end

# ------------------------------------------------------------
# Physics-only baseline (Eq2 uses β*Pgen - β*Pload)
# ------------------------------------------------------------

function physics_rhs(params::Vector{Float64}, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        up_t, um_t = sdf.u_plus[idx], sdf.u_minus[idx]
        Ipos_t, Ineg_t = sdf.I_u_pos[idx], sdf.I_u_neg[idx]
        d_t = sdf.d[idx]
        Pgen_t, Pload_t = sdf.Pgen[idx], sdf.Pload[idx]
        ηin, ηout, α, β, γ = params[1:5]
        du[1] = ηin * up_t * Ipos_t - (1/ηout) * um_t * Ineg_t - d_t
        du[2] = -α * x2 + β * Pgen_t - β * Pload_t + γ * x1
    end
    return rhs!
end

# ------------------------------------------------------------
# BNode loading and functions (Objective 1)
# ------------------------------------------------------------

function build_bnode_theta(width::Int)
    W1 = width * 4  # Eq1: x1,x2,u,d
    W2 = width * 4  # Eq2: x1,x2,Pgen,Pload
    b1 = width
    b2 = width
    return W1 + W2 + b1 + b2
end

function ftheta1_bnode(x1::Float64, x2::Float64, u::Float64, d::Float64, θ::Vector{Float64}, width::Int)
    start_idx = 1
    W1 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
    b1 = θ[start_idx+width*4:start_idx+width*4+width-1]
    inputs = [x1, x2, u, d]
    h = tanh.(W1 * inputs .+ b1)
    return sum(h)
end

function ftheta2_bnode(x1::Float64, x2::Float64, Pgen::Float64, Pload::Float64, θ::Vector{Float64}, width::Int)
    start_idx = 1 + width*4 + width
    W2 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
    b2 = θ[start_idx+width*4:start_idx+width*4+width-1]
    inputs = [x1, x2, Pgen, Pload]
    h = tanh.(W2 * inputs .+ b2)
    return sum(h)
end

function bnode_rhs(θ::Vector{Float64}, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        u_t = sdf.u[idx]
        d_t = sdf.d[idx]
        Pgen_t = sdf.Pgen[idx]
        Pload_t = sdf.Pload[idx]
        du[1] = ftheta1_bnode(x1, x2, u_t, d_t, θ, width)
        du[2] = ftheta2_bnode(x1, x2, Pgen_t, Pload_t, θ, width)
    end
    return rhs!
end

function load_bnode_posterior_mean()
    bnode_file = joinpath(ckpt_dir, "bnode_posterior.bson")
    @assert isfile(bnode_file) "Missing checkpoints/bnode_posterior.bson"
    data = BSON.load(bnode_file)
    @assert haskey(data, :chain) && haskey(data, :width)
    chain = data[:chain]
    width = data[:width]
    arr = Array(chain)  # dims: iterations × params × chains
    nparams = build_bnode_theta(width)
    @assert size(arr, 2) >= nparams+1  # expect σ as last
    θ_mean = vec(mean(arr[:, 1:nparams, :], dims=(1,3)))
    return θ_mean, width
end

# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------

function solve_model(rhs!, T, Y)
    x0 = Y[1, :]
    prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
    sol = solve(prob, Tsit5(); saveat=T, abstol=1e-6, reltol=1e-6)
    if sol.retcode != :Success
        return nothing
    end
    return reduce(hcat, (sol(t) for t in T))'
end

function metrics_for(Yhat, Y)
    rmse1 = sqrt(mean((Yhat[:,1] .- Y[:,1]).^2))
    rmse2 = sqrt(mean((Yhat[:,2] .- Y[:,2]).^2))
    r21 = 1 - sum((Yhat[:,1] .- Y[:,1]).^2) / sum((Y[:,1] .- mean(Y[:,1])).^2)
    r22 = 1 - sum((Yhat[:,2] .- Y[:,2]).^2) / sum((Y[:,2] .- mean(Y[:,2])).^2)
    mae1 = mean(abs.(Yhat[:,1] .- Y[:,1]))
    mae2 = mean(abs.(Yhat[:,2] .- Y[:,2]))
    return (rmse1=rmse1, rmse2=rmse2, r2_x1=r21, r2_x2=r22, mae1=mae1, mae2=mae2)
end

function bootstrap_ci(v::AbstractVector{<:Real}; B::Int=500)
    rng = MersenneTwister(42)
    n = length(v)
    bs = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(rng, 1:n, n)
        bs[b] = mean(v[idx])
    end
    return mean(v), quantile(bs, 0.025), quantile(bs, 0.975)
end

# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------

models = ["physics", "ude", "bnode"]
rows = Vector{NamedTuple}()

θ_bnode_mean = nothing
width_bnode = nothing
if isfile(joinpath(ckpt_dir, "bnode_posterior.bson"))
    try
        θ_bnode_mean, width_bnode = load_bnode_posterior_mean()
    catch e
        @warn "Failed to load BNode posterior mean: $e"
    end
end

for (sid, sdf) in test_sc
    T = Vector{Float64}(sdf.time)
    Y = Matrix(sdf[:, [:x1, :x2]])

    # Physics-only: use UDE physics params p[1:5]
    phys_rhs! = physics_rhs(ude_params[1:5], T, sdf)
    Yhat_phys = solve_model(phys_rhs!, T, Y)
    if Yhat_phys !== nothing
        m = metrics_for(Yhat_phys, Y)
        push!(rows, (scenario=sid, model="physics", rmse_x1=m.rmse1, rmse_x2=m.rmse2, r2_x1=m.r2_x1, r2_x2=m.r2_x2, mae_x1=m.mae1, mae_x2=m.mae2))
    end

    # UDE
    ude_rhs! = ude_rhs(ude_params, ude_width, T, sdf)
    Yhat_ude = solve_model(ude_rhs!, T, Y)
    if Yhat_ude !== nothing
        m = metrics_for(Yhat_ude, Y)
        push!(rows, (scenario=sid, model="ude", rmse_x1=m.rmse1, rmse_x2=m.rmse2, r2_x1=m.r2_x1, r2_x2=m.r2_x2, mae_x1=m.mae1, mae_x2=m.mae2))
    end

    # BNode (if available)
    if θ_bnode_mean !== nothing
        bnode_rhs! = bnode_rhs(θ_bnode_mean, width_bnode, T, sdf)
        Yhat_b = solve_model(bnode_rhs!, T, Y)
        if Yhat_b !== nothing
            m = metrics_for(Yhat_b, Y)
            push!(rows, (scenario=sid, model="bnode", rmse_x1=m.rmse1, rmse_x2=m.rmse2, r2_x1=m.r2_x1, r2_x2=m.r2_x2, mae_x1=m.mae1, mae_x2=m.mae2))
        end
    end
end

metrics_df = DataFrame(rows)
CSV.write(joinpath(res_dir, "comprehensive_metrics.csv"), metrics_df)

# Aggregate per model with CIs
models_present = unique(metrics_df.model)
open(joinpath(res_dir, "comprehensive_comparison_summary.md"), "w") do io
    write(io, "# Comprehensive Model Comparison (Per-Scenario)\n\n")
    write(io, "Test scenarios: $(length(test_sc))\n\n")
    for mdl in models_present
        sub = metrics_df[metrics_df.model .== mdl, :]
        write(io, "## $(mdl)\n\n")
        for col in [:rmse_x1, :rmse_x2, :r2_x1, :r2_x2, :mae_x1, :mae_x2]
            μ, lo, hi = bootstrap_ci(Vector{Float64}(sub[!, col]))
            write(io, "- $(String(col)): $(round(μ,digits=4)) [$(round(lo,digits=4)), $(round(hi,digits=4))]\n")
        end
        write(io, "\n")
    end
end

println("✅ Wrote results/comprehensive_metrics.csv and results/comprehensive_comparison_summary.md")

# ------------------------------------------------------------
# Symbolic extraction for UDE's fθ(Pgen)
# ------------------------------------------------------------

function polyfit_xy(x::Vector{Float64}, y::Vector{Float64}, deg::Int)
    n = length(x)
    X = zeros(n, deg+1)
    for i in 0:deg
        X[:, i+1] .= x .^ i
    end
    # Least squares
    c = X \ y
    return c  # coefficients for ∑ c[i+1] * x^i
end

function polyval_xy(c::Vector{Float64}, x::Vector{Float64})
    deg = length(c) - 1
    y = zeros(length(x))
    for i in 0:deg
        y .+= c[i+1] .* (x .^ i)
    end
    return y
end

# Build Pgen grid across the test set
Pgen_all = Vector{Float64}(test_df.Pgen)
qlo, qhi = quantile(Pgen_all, 0.01), quantile(Pgen_all, 0.99)
grid = range(qlo, qhi; length=200) |> collect

θ_ude = ude_params[6:end]
fvals = [ftheta_ude(pg, θ_ude, ude_width) for pg in grid]

deg = 3
c = polyfit_xy(grid, fvals, deg)

open(joinpath(res_dir, "ude_symbolic_extraction.md"), "w") do io
    write(io, "# UDE Symbolic Extraction\n\n")
    write(io, "Fitted polynomial (degree $(deg)) for fθ(Pgen):\\n\n")
    # Pretty print: c0 + c1 P + c2 P^2 + c3 P^3
    terms = ["$(round(c[i],digits=6)) * Pgen^$(i-1)" for i in 1:length(c)]
    write(io, "fθ(Pgen) ≈ " * join(terms, " + ") * "\n\n")
end

println("✅ Wrote results/ude_symbolic_extraction.md")


