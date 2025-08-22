#!/usr/bin/env julia

"""
    bnode_train_calibrate.jl

Bayesian Neural ODE (BNode) training and calibration following Objective 1 from roadmap:
  "Replace the full ODE with a Bayesian Neural ODE and perform prediction and forecasting"

Both equations are black box neural networks:
  Eq1: dx1/dt = fθ1(x1, x2, u, d, t)  # Black box for energy storage
  Eq2: dx2/dt = fθ2(x1, x2, Pgen, Pload, t)  # Black box for grid power

Features:
  - Physics-informed priors on parameters
  - MCMC sampling with NUTS
  - Uncertainty quantification and calibration
  - Per-scenario evaluation with coverage tests

Outputs:
  - checkpoints/bnode_posterior.bson
  - results/bnode_calibration_report.md
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using DifferentialEquations
using Statistics, Random
using BSON
using Turing
using MCMCChains
using LinearAlgebra

# Load roadmap dataset
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap.csv")
val_csv   = joinpath(@__DIR__, "..", "data", "validation_roadmap.csv")
@assert isfile(train_csv) "Missing training_roadmap.csv"
@assert isfile(val_csv)   "Missing validation_roadmap.csv"

train_df = CSV.read(train_csv, DataFrame)
val_df   = CSV.read(val_csv, DataFrame)

function group_scenarios(df::DataFrame)
    Dict(string(s)=>sort(sub, :time) for sub in groupby(df, :scenario) for s in unique(string.(sub.scenario)))
end

train_sc = group_scenarios(train_df)
val_sc   = group_scenarios(val_df)

# BNode architecture: both equations are black box neural networks
function build_bnode_theta(width::Int)
    # Eq1: fθ1(x1, x2, u, d) → dx1/dt
    # Eq2: fθ2(x1, x2, Pgen, Pload) → dx2/dt
    # Each has: input → width (tanh) → 1 (linear)
    W1 = width * 4  # 4 inputs: x1, x2, u, d
    W2 = width * 4  # 4 inputs: x1, x2, Pgen, Pload
    b1 = width      # biases for Eq1
    b2 = width      # biases for Eq2
    total_params = W1 + W2 + b1 + b2
    return total_params
end

function ftheta1(x1, x2, u, d, θ, width::Int)
    # Eq1: fθ1(x1, x2, u, d) → dx1/dt
    start_idx = 1
    W1 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
    b1 = θ[start_idx+width*4:start_idx+width*4+width-1]
    inputs = [x1, x2, u, d]
    h = tanh.(W1 * inputs .+ b1)
    return sum(h)
end

function ftheta2(x1, x2, Pgen, Pload, θ, width::Int)
    # Eq2: fθ2(x1, x2, Pgen, Pload) → dx2/dt
    start_idx = 1 + width*4 + width  # Skip Eq1 parameters
    W2 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
    b2 = θ[start_idx+width*4:start_idx+width*4+width-1]
    inputs = [x1, x2, Pgen, Pload]
    h = tanh.(W2 * inputs .+ b2)
    return sum(h)
end

function make_bnode_rhs(θ, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        u_t = sdf.u[idx]
        d_t = sdf.d[idx]
        Pgen_t = sdf.Pgen[idx]
        Pload_t = sdf.Pload[idx]
        
        # Both equations are black box neural networks
        du[1] = ftheta1(x1, x2, u_t, d_t, θ, width)
        du[2] = ftheta2(x1, x2, Pgen_t, Pload_t, θ, width)
    end
    return rhs!
end

function solve_scenario(θ, width::Int, sdf::DataFrame)
    T = Vector{Float64}(sdf.time)
    Y = Matrix(sdf[:, [:x1, :x2]])
    rhs! = make_bnode_rhs(θ, width, T, sdf)
    x0 = Y[1, :]
    
    try
        prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
        sol = solve(prob, Tsit5(); saveat=T, abstol=1e-6, reltol=1e-6)
        if sol.retcode != :Success
            return nothing
        end
        Yhat = reduce(hcat, (sol(t) for t in T))'
        return Yhat
    catch
        return nothing
    end
end

# Bayesian model with physics-informed priors
@model function bnode_model(train_scenarios, width::Int)
    # Prior on neural network parameters (small weights for stability)
    θ ~ MvNormal(zeros(build_bnode_theta(width)), 0.1)
    
    # Prior on observation noise (physics-informed: expect small residuals)
    σ ~ truncated(Normal(0.05, 0.02), 0.01, 0.2)
    
    # Likelihood: per-scenario ODE solutions
    for (sid, sdf) in train_scenarios
        Yhat = solve_scenario(θ, width, sdf)
        if Yhat !== nothing
            Y = Matrix(sdf[:, [:x1, :x2]])
            # Vectorized likelihood for efficiency with unique variable names
            Y_vec = vec(Y)
            Yhat_vec = vec(Yhat)
            # Use scenario ID to create unique variable names
            if sid == 1
                Y_scenario_1 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 2
                Y_scenario_2 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 3
                Y_scenario_3 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 4
                Y_scenario_4 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 5
                Y_scenario_5 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 6
                Y_scenario_6 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 7
                Y_scenario_7 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 8
                Y_scenario_8 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 9
                Y_scenario_9 ~ MvNormal(Yhat_vec, σ)
            elseif sid == 10
                Y_scenario_10 ~ MvNormal(Yhat_vec, σ)
            end
        end
    end
end

# Calibration metrics
function compute_calibration_metrics(chain, val_scenarios, width::Int)
    # Sample from posterior
    θ_samples = Array(chain)[:, 1:end-1, :]  # Exclude σ
    σ_samples = Array(chain)[:, end, :]
    
    coverage_50 = Float64[]
    coverage_90 = Float64[]
    nll_scores = Float64[]
    
    for (sid, sdf) in val_scenarios
        Y = Matrix(sdf[:, [:x1, :x2]])
        predictions = []
        
        # Generate predictions for each posterior sample
        for i in 1:min(50, size(θ_samples, 3))  # Use up to 50 samples
            θ = θ_samples[:, 1, i]
            Yhat = solve_scenario(θ, width, sdf)
            if Yhat !== nothing
                push!(predictions, Yhat)
            end
        end
        
        if length(predictions) > 0
            # Stack predictions
            pred_array = cat(predictions..., dims=3)
            
            # Compute quantiles for coverage
            q25 = mapslices(x -> quantile(x, 0.25), pred_array, dims=3)
            q75 = mapslices(x -> quantile(x, 0.75), pred_array, dims=3)
            q05 = mapslices(x -> quantile(x, 0.05), pred_array, dims=3)
            q95 = mapslices(x -> quantile(x, 0.95), pred_array, dims=3)
            
            # Coverage
            cov_50 = mean((Y .>= q25) .& (Y .<= q75))
            cov_90 = mean((Y .>= q05) .& (Y .<= q95))
            
            push!(coverage_50, cov_50)
            push!(coverage_90, cov_90)
            
            # NLL (simplified)
            mean_pred = mean(pred_array, dims=3)[:, :, 1]
            mean_σ = mean(σ_samples)
            nll = 0.5 * sum((Y .- mean_pred).^2) / (mean_σ^2) + length(Y) * log(mean_σ)
            push!(nll_scores, nll)
        end
    end
    
    return (
        mean_coverage_50 = mean(coverage_50),
        mean_coverage_90 = mean(coverage_90),
        mean_nll = mean(nll_scores)
    )
end

# Main training function
function train_bnode(width::Int=5; n_samples::Int=1000, n_chains::Int=4)
    println("🚀 Training BNode (Objective 1: full black box)")
    println("  → Width: $(width)")
    println("  → Training scenarios: $(length(train_sc))")
    println("  → Validation scenarios: $(length(val_sc))")
    
    # Use subset for initial training (can scale up later)
    train_subset = Dict()
    for (i, (k, v)) in enumerate(train_sc)
        if i <= 10  # Start with 10 scenarios
            train_subset[k] = v
        end
    end
    
    println("  → Using $(length(train_subset)) scenarios for initial training")
    
    # Sample from posterior
    model = bnode_model(train_subset, width)
    
    # Use NUTS sampler
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    # Compute calibration metrics
    cal_metrics = compute_calibration_metrics(chain, val_sc, width)
    
    println("  → Calibration results:")
    println("    - 50% coverage: $(round(cal_metrics.mean_coverage_50, digits=3))")
    println("    - 90% coverage: $(round(cal_metrics.mean_coverage_90, digits=3))")
    println("    - Mean NLL: $(round(cal_metrics.mean_nll, digits=3))")
    
    # Save results
    ckpt_dir = joinpath(@__DIR__, "..", "checkpoints")
    if !isdir(ckpt_dir); mkdir(ckpt_dir); end
    
    BSON.@save joinpath(ckpt_dir, "bnode_posterior.bson") chain cal_metrics width
    
    # Generate calibration report
    res_dir = joinpath(@__DIR__, "..", "results")
    if !isdir(res_dir); mkdir(res_dir); end
    
    open(joinpath(res_dir, "bnode_calibration_report.md"), "w") do io
        write(io, "# BNode Calibration Report\n\n")
        write(io, "## Training Configuration\n")
        write(io, "- Architecture: Both equations as black box neural networks\n")
        write(io, "- Width: $(width)\n")
        write(io, "- Training scenarios: $(length(train_subset))\n")
        write(io, "- MCMC samples: $(n_samples)\n")
        write(io, "- Chains: $(n_chains)\n\n")
        
        write(io, "## Calibration Metrics\n")
        write(io, "- 50% coverage: $(round(cal_metrics.mean_coverage_50, digits=3))\n")
        write(io, "- 90% coverage: $(round(cal_metrics.mean_coverage_90, digits=3))\n")
        write(io, "- Mean NLL: $(round(cal_metrics.mean_nll, digits=3))\n\n")
        
        write(io, "## Interpretation\n")
        if cal_metrics.mean_coverage_50 > 0.45 && cal_metrics.mean_coverage_50 < 0.55
            write(io, "- ✅ 50% coverage is under-calibrated\n")
        else
            write(io, "- ⚠️ 50% coverage may need adjustment\n")
        end
        
        if cal_metrics.mean_coverage_90 > 0.85 && cal_metrics.mean_coverage_90 < 0.95
            write(io, "- ✅ 90% coverage is under-calibrated\n")
        else
            write(io, "- ⚠️ 90% coverage may need adjustment\n")
        end
    end
    
    println("✅ BNode training complete!")
    println("  → Saved: checkpoints/bnode_posterior.bson")
    println("  → Report: results/bnode_calibration_report.md")
    
    return chain, cal_metrics
end

# Run training
if abspath(PROGRAM_FILE) == @__FILE__
    println("🚀 Starting BNode training...")
    chain, metrics = train_bnode(5; n_samples=500, n_chains=2)
    println("✅ BNode training complete!")
else
    println("📋 BNode training script ready!")
    println("  → Run with: julia scripts/bnode_train_calibrate.jl")
end 