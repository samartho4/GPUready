#!/usr/bin/env julia

"""
    evaluate_per_scenario.jl

Per-scenario evaluation of roadmap-aligned UDE model with bootstrap confidence intervals.
Loads checkpoint from scripts/train_roadmap_models.jl or scripts/tune_ude_hparams.jl.
Outputs: results/perscenario_summary.md with CIs, results/perscenario_metrics.csv
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using DifferentialEquations
using Statistics, Random
using BSON

# Load test data
test_csv = joinpath(@__DIR__, "..", "data", "test_roadmap.csv")
@assert isfile(test_csv) "Missing test_roadmap.csv"
test_df = CSV.read(test_csv, DataFrame)

function group_scenarios(df::DataFrame)
    Dict(string(s)=>sort(sub, :time) for sub in groupby(df, :scenario) for s in unique(string.(sub.scenario)))
end

test_sc = group_scenarios(test_df)

# Load UDE checkpoint (try both possible sources)
ckpt_dir = joinpath(@__DIR__, "..", "checkpoints")
ude_ckpt = joinpath(ckpt_dir, "ude_best_tuned.bson")
if !isfile(ude_ckpt)
    ude_ckpt = joinpath(ckpt_dir, "ude_roadmap_opt.bson")
end
@assert isfile(ude_ckpt) "No UDE checkpoint found"

ckpt_data = BSON.load(ude_ckpt)
if haskey(ckpt_data, :best_ckpt)
    # From tuning
    params = ckpt_data[:best_ckpt][:p]
    width = ckpt_data[:best_ckpt][:width]
else
    # From direct training
    params = ckpt_data[:ude_params_opt]
    width = 5  # default
end

println("📊 Loading UDE params: $(length(params)) parameters")

# Neural network: fθ(Pgen) → ℝ - CORRECTED per SS
function ftheta(Pgen::Float64, θ::Vector{Float64}, width::Int)
    W1 = reshape(θ[1:width], width, 1)  # width x 1 matrix
    b1 = θ[width+1:width+width]  # width biases
    h = tanh.(W1 * [Pgen] .+ b1)
    return sum(h)
end

# RHS builder - CORRECTED per SS
function make_rhs(params::Vector{Float64}, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        up_t, um_t = sdf.u_plus[idx], sdf.u_minus[idx]
        Ipos_t, Ineg_t = sdf.I_u_pos[idx], sdf.I_u_neg[idx]
        d_t = sdf.d[idx]
        Pgen_t, Pload_t = sdf.Pgen[idx], sdf.Pload[idx]
        ηin, ηout, α, β, γ = params[1:5]  # ADDED β
        θ = params[6:end]  # θ starts at index 6
        du[1] = ηin * up_t * Ipos_t - (1/ηout) * um_t * Ineg_t - d_t
        du[2] = -α * x2 + ftheta(Pgen_t, θ, width) - β * Pload_t + γ * x1  # CORRECTED per SS
    end
    return rhs!
end

# Per-scenario evaluation
function eval_scenario(params, width::Int, sdf::DataFrame)
    T = Vector{Float64}(sdf.time)
    Y = Matrix(sdf[:, [:x1, :x2]])
    rhs! = make_rhs(params, width, T, sdf)
    x0 = Y[1, :]
    prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
    sol = solve(prob, Tsit5(); saveat=T, abstol=1e-6, reltol=1e-6)
    if sol.retcode != :Success
        return (Inf, Inf, -Inf, -Inf, Inf, Inf)
    end
    Yhat = reduce(hcat, (sol(t) for t in T))'
    rmse1 = sqrt(mean((Yhat[:,1] .- Y[:,1]).^2))
    rmse2 = sqrt(mean((Yhat[:,2] .- Y[:,2]).^2))
    r21 = 1 - sum((Yhat[:,1] .- Y[:,1]).^2) / sum((Y[:,1] .- mean(Y[:,1])).^2)
    r22 = 1 - sum((Yhat[:,2] .- Y[:,2]).^2) / sum((Y[:,2] .- mean(Y[:,2])).^2)
    mae1 = mean(abs.(Yhat[:,1] .- Y[:,1]))
    mae2 = mean(abs.(Yhat[:,2] .- Y[:,2]))
    return (rmse1, rmse2, r21, r22, mae1, mae2)
end

# Bootstrap confidence intervals
function bootstrap_ci(v, B=1000)
    n = length(v)
    bs = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(1:n, n)
        bs[b] = mean(v[idx])
    end
    return mean(v), quantile(bs, 0.025), quantile(bs, 0.975)
end

# Evaluate all scenarios
results = DataFrame(scenario=String[], rmse_x1=Float64[], rmse_x2=Float64[], 
                   r2_x1=Float64[], r2_x2=Float64[], mae_x1=Float64[], mae_x2=Float64[])

for (sid, sdf) in test_sc
    r1, r2, R1, R2, m1, m2 = eval_scenario(params, width, sdf)
    push!(results, (sid, r1, r2, R1, R2, m1, m2))
end

# Aggregate with CIs
metrics = Dict{Symbol, Tuple{Float64, Float64, Float64}}()
for col in [:rmse_x1, :rmse_x2, :r2_x1, :r2_x2, :mae_x1, :mae_x2]
    μ, lo, hi = bootstrap_ci(Vector{Float64}(results[!, col]))
    metrics[col] = (μ, lo, hi)
end

# Save detailed results
res_dir = joinpath(@__DIR__, "..", "results")
if !isdir(res_dir); mkdir(res_dir); end
CSV.write(joinpath(res_dir, "perscenario_metrics.csv"), results)

# Generate summary report
open(joinpath(res_dir, "perscenario_summary.md"), "w") do io
    write(io, "# Per-Scenario UDE Evaluation Summary\n\n")
    write(io, "**Test Set:** $(length(test_sc)) scenarios\n\n")
    write(io, "## Performance Metrics (95% Bootstrap CI)\n\n")
    write(io, "| Metric | Mean | 95% CI |\n")
    write(io, "|--------|------|--------|\n")
    for (metric, (μ, lo, hi)) in metrics
        write(io, "| $(metric) | $(round(μ, digits=4)) | [$(round(lo, digits=4)), $(round(hi, digits=4))] |\n")
    end
    write(io, "\n## Physics Parameters\n\n")
    ηin, ηout, α, β, γ = params[1:5]
    write(io, "- ηin (charging efficiency): $(round(ηin, digits=4))\n")
    write(io, "- ηout (discharging efficiency): $(round(ηout, digits=4))\n")
    write(io, "- α (grid damping): $(round(α, digits=4))\n")
    write(io, "- β (load coupling): $(round(β, digits=4))\n")
    write(io, "- γ (storage-grid coupling): $(round(γ, digits=4))\n")
end

println("✅ Evaluation complete → results/perscenario_summary.md") 