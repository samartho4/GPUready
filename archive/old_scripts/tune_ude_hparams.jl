#!/usr/bin/env julia

"""
    tune_ude_hparams.jl

Grid-search UDE tuning on roadmap dataset with multiple seeds and per-scenario evaluation.
- Physics-only Eq1: dx1/dt = ηin * u+(t) - (1/ηout) * u−(t) - d(t)
- UDE Eq2: dx2/dt = -α x2 + fθ(Pgen) - β*Pload + γ * x1 (NN replaces only β*Pgen term)
- Grid over: width ∈ {4,5,6}, weight decay λ ∈ {1e-5,1e-4,5e-4,1e-3}, reltol ∈ {1e-5,1e-6}
- Seeds: 5
- Reports mean±CI for RMSE/R² (x1,x2) across scenarios; saves best checkpoint
- Outputs: results/ude_tuning_results.csv, results/ude_tuning_summary.md
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using DifferentialEquations
using Statistics, Random
using BSON
using Optim

# Load data
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

# Model components - CORRECTED per SS: fθ(Pgen) replaces β*Pgen only
function build_theta(width::Int)
    # Return index map and init θ for fθ(Pgen) only
    W = width
    θ = 0.1 .* randn(W + W)  # W*1 + W (weights + biases)
    return θ
end

function ftheta(Pgen::Float64, θ::Vector{Float64}, width::Int)
    W1 = reshape(θ[1:width], width, 1)  # width x 1 matrix
    b1 = θ[width+1:width+width]  # width biases
    h = tanh.(W1 * [Pgen] .+ b1)
    return sum(h)
end

function make_rhs(params::Vector{Float64}, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        up_t, um_t = sdf.u_plus[idx], sdf.u_minus[idx]
        Ipos_t, Ineg_t = sdf.I_u_pos[idx], sdf.I_u_neg[idx]
        d_t = sdf.d[idx]
        Pgen_t, Pload_t = sdf.Pgen[idx], sdf.Pload[idx]
        ηin, ηout, α, β, γ = params[1:5]  # Physics parameters
        θ = params[6:end]  # θ starts at index 6
        # Eq1: Physics-only (energy storage)
        du[1] = ηin * up_t * Ipos_t - (1/ηout) * um_t * Ineg_t - d_t
        # Eq2: UDE - fθ(Pgen) replaces β*Pgen, keep β*Pload separate
        du[2] = -α * x2 + ftheta(Pgen_t, θ, width) - β * Pload_t + γ * x1
    end
    return rhs!
end

function eval_scenario(params, width::Int, reltol::Float64, sdf::DataFrame)
    try
        T = Vector{Float64}(sdf.time)
        Y = Matrix(sdf[:, [:x1, :x2]])
        rhs! = make_rhs(params, width, T, sdf)
        x0 = Y[1, :]
        
        # Check initial conditions
        if any(isnan.(x0)) || any(isinf.(x0))
            return (Inf, Inf, -Inf, -Inf)
        end
        
        prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
        sol = solve(prob, Rosenbrock23(); saveat=T, abstol=reltol*0.1, reltol=reltol, maxiters=20000)
        
        if sol.retcode != :Success
            return (Inf, Inf, -Inf, -Inf)
        end
        
        Yhat = reduce(hcat, (sol(t) for t in T))'
        
        # Check solution validity
        if any(isnan.(Yhat)) || any(isinf.(Yhat))
            return (Inf, Inf, -Inf, -Inf)
        end
        
        rmse1 = sqrt(mean((Yhat[:,1] .- Y[:,1]).^2))
        rmse2 = sqrt(mean((Yhat[:,2] .- Y[:,2]).^2))
        r21 = 1 - sum((Yhat[:,1] .- Y[:,1]).^2) / sum((Y[:,1] .- mean(Y[:,1])).^2)
        r22 = 1 - sum((Yhat[:,2] .- Y[:,2]).^2) / sum((Y[:,2] .- mean(Y[:,2])).^2)
        
        # Check for invalid metrics
        if any(isnan.([rmse1, rmse2, r21, r22])) || any(isinf.([rmse1, rmse2, r21, r22]))
            return (Inf, Inf, -Inf, -Inf)
        end
        
        return (rmse1, rmse2, r21, r22)
    catch e
        return (Inf, Inf, -Inf, -Inf)
    end
end

function constrain!(p)
    p[1] = clamp(p[1], 0.85, 0.98)  # ηin
    p[2] = clamp(p[2], 0.85, 0.98)  # ηout
    p[3] = clamp(p[3], 0.03, 0.2)   # α
    p[4] = clamp(p[4], 0.5, 2.0)    # β (ADDED)
    p[5] = clamp(p[5], 0.005, 0.05) # γ
    return p
end

function total_loss(p, width::Int, reltol::Float64, scdict)
    losses = Float64[]
    for sdf in values(scdict)
        try
            T = Vector{Float64}(sdf.time)
            Y = Matrix(sdf[:, [:x1, :x2]])
            rhs! = make_rhs(p, width, T, sdf)
            x0 = Y[1, :]
            
            # Add bounds checking for initial conditions
            if any(isnan.(x0)) || any(isinf.(x0))
                push!(losses, 1e6)
                continue
            end
            
            # Try to solve ODE with error handling
            prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
            sol = solve(prob, Rosenbrock23(); saveat=T, abstol=reltol*0.1, reltol=reltol, maxiters=500)
            
            if sol.retcode != :Success
                push!(losses, 1e6)
                continue
            end
            
            # Check if solution is valid
            Yhat = reduce(hcat, (sol(t) for t in T))'
            if any(isnan.(Yhat)) || any(isinf.(Yhat))
                push!(losses, 1e6)
                continue
            end
            
            loss = mean((Yhat .- Y).^2)
            if isnan(loss) || isinf(loss)
                push!(losses, 1e6)
            else
                push!(losses, loss)
            end
            
        catch e
            # Catch any other errors and assign high loss
            push!(losses, 1e6)
        end
    end
    
    # Return mean loss, but handle empty case
    if isempty(losses)
        return 1e6
    end
    return mean(losses)
end

function optimize_once(width::Int, λ::Float64, reltol::Float64, seed::Int)
    Random.seed!(seed)
    # params = [ηin, ηout, α, β, γ, θ...] - UPDATED structure
    θ_size = width + width  # weights + biases for fθ(Pgen)
    
    # Better initialization with smaller neural network weights
    p = vcat([0.9, 0.9, 0.1, 1.0, 0.02], 0.01 .* randn(θ_size))  # Smaller initial weights
    constrain!(p)
    
    # simple LBFGS with weight decay on θ
    function obj(vecp)
        try
            q = copy(vecp); constrain!(q)
            loss = total_loss(q, width, reltol, train_sc)
            if isnan(loss) || isinf(loss)
                return 1e6
            end
            reg = λ * sum(q[6:end].^2)  # θ starts at index 6
            return loss + reg
        catch e
            return 1e6  # Return high loss on any error
        end
    end
    
    # More conservative optimization settings
    res = Optim.optimize(obj, p, Optim.LBFGS(), 
                        Optim.Options(g_tol=1e-3, x_abstol=1e-3, f_reltol=1e-3, iterations=50))
    
    bestp = Optim.minimizer(res)
    constrain!(bestp)
    
    # evaluate on val per-scenario
    metrics = DataFrame(scenario=String[], rmse_x1=Float64[], rmse_x2=Float64[], r2_x1=Float64[], r2_x2=Float64[])
    for (sid, sdf) in val_sc
        r1, r2, R1, R2 = eval_scenario(bestp, width, reltol, sdf)
        push!(metrics, (sid, r1, r2, R1, R2))
    end
    return bestp, metrics
end

function agg_metrics(df::DataFrame)
    function ci(v)
        B=200; rng=MersenneTwister(42); n=length(v); bs=Vector{Float64}(undef,B)
        for b in 1:B
            idx = rand(rng, 1:n, n)
            bs[b] = mean(v[idx])
        end
        return mean(v), quantile(bs,0.025), quantile(bs,0.975)
    end
    m = Dict{Symbol,Tuple{Float64,Float64,Float64}}()
    for col in [:rmse_x1,:rmse_x2,:r2_x1,:r2_x2]
        μ,lo,hi = ci(Vector{Float64}(df[!,col])); m[col]=(μ,lo,hi)
    end
    return m
end

widths = [4,5,6]
λs = [1e-5,1e-4,5e-4,1e-3]
reltols = [1e-5,1e-6]
seeds = 1:5

results = DataFrame(width=Int[], lambda=Float64[], reltol=Float64[], seed=Int[],
    mean_rmse_x1=Float64[], mean_rmse_x2=Float64[], mean_r2_x1=Float64[], mean_r2_x2=Float64[])

global best_score = Inf
global best_ckpt = nothing
global best_cfg = nothing

for w in widths, λ in λs, rt in reltols
    for s in seeds
        p, met = optimize_once(w, λ, rt, s)
        m = agg_metrics(met)
        push!(results, (w, λ, rt, s, m[:rmse_x1][1], m[:rmse_x2][1], m[:r2_x1][1], m[:r2_x2][1]))
        # selection: minimize rmse_x2 + 0.2*rmse_x1
        score = m[:rmse_x2][1] + 0.2*m[:rmse_x1][1]
        if score < best_score
            global best_score = score
            global best_ckpt = (p=p, width=w, lambda=λ, reltol=rt)
            global best_cfg = (w,λ,rt,s)
        end
    end
end

res_dir = joinpath(@__DIR__, "..", "results")
if !isdir(res_dir); mkdir(res_dir); end
CSV.write(joinpath(res_dir,"ude_tuning_results.csv"), results)
open(joinpath(res_dir,"ude_tuning_summary.md"),"w") do io
    write(io, "# UDE Tuning Summary\n\n")
    if best_cfg === nothing
        write(io, "No valid configuration found during grid search. Review data quality and solver tolerances.\n")
    else
        write(io, "Best config (seed=$(best_cfg[4])): width=$(best_cfg[1]), lambda=$(best_cfg[2]), reltol=$(best_cfg[3])\n")
        write(io, "Selection score (rmse_x2+0.2*rmse_x1)=$(round(best_score,digits=6))\n")
    end
end

# Save best checkpoint for evaluation
ckpt_dir = joinpath(@__DIR__, "..", "checkpoints")
if !isdir(ckpt_dir); mkdir(ckpt_dir); end
BSON.@save joinpath(ckpt_dir, "ude_best_tuned.bson") best_ckpt
println("✅ Saved tuning results and best checkpoint → results/ and checkpoints/ude_best_tuned.bson") 