module Training

export train!

using Random, TOML, Dates, BSON, CSV, DataFrames, Statistics
using DifferentialEquations
using Turing
using MCMCChains
using Pkg

include(joinpath(@__DIR__, "microgrid_system.jl"))
include(joinpath(@__DIR__, "neural_ode_architectures.jl"))
using .NeuralNODEArchitectures
using .Microgrid

# ----------------------
# Helpers
# ----------------------
const CONFIG_PATH = joinpath(@__DIR__, "..", "config", "config.toml")

function load_config()
    if isfile(CONFIG_PATH)
        try
            return TOML.parsefile(CONFIG_PATH)
        catch e
            @warn "Failed to parse config" error=e path=CONFIG_PATH
        end
    end
    return Dict{String,Any}()
end

getcfg(config::Dict, dflt, ks...) = begin
    v = config
    for k in ks
        if v isa Dict && haskey(v, String(k))
            v = v[String(k)]
        else
            return dflt
        end
    end
    return v
end

function pick_arch(arch::AbstractString)
    a = lowercase(String(arch))
    if a == "baseline"
        return (:baseline, baseline_nn!, 10)
    elseif a == "baseline_bias"
        return (:baseline_bias, baseline_nn_bias!, 14)
    elseif a == "deep"
        return (:deep, deep_nn!, 26)
    else
        @warn "Unknown arch, defaulting to baseline" arch
        return (:baseline, baseline_nn!, 10)
    end
end

function capture_metadata(config::Dict{String,Any})
    git_sha = try
        readchomp(`git rev-parse HEAD`)
    catch
        "unknown"
    end
    
    # Get package versions using Pkg
    pkg_deps = Pkg.dependencies()
    packages = Dict{String, String}()
    for (uuid, dep) in pkg_deps
        packages[string(dep.name)] = string(dep.version)
    end
    
    metadata = Dict{Symbol,Any}(
        :git_sha => git_sha,
        :julia_version => string(VERSION),
        :os => Sys.KERNEL,
        :machine => Sys.MACHINE,
        :cpu => Sys.CPU_NAME,
        :packages => packages,
        :config => config,
        :timestamp => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    )
    
    # Save to paper/results/run_meta.toml for reproducibility
    results_dir = joinpath(@__DIR__, "..", "paper", "results")
    mkpath(results_dir)
    
    # Convert symbols and other non-TOML types for serialization
    metadata_sanitized = sanitize_for_toml(metadata)
    
    open(joinpath(results_dir, "run_meta.toml"), "w") do io
        TOML.print(io, metadata_sanitized)
    end
    
    return metadata
end

function initial_theta(init_scheme::AbstractString, init_scale::Float64, dim::Int)
    if init_scheme == "zeros"
        return zeros(dim)
    else
        return init_scale .* randn(dim)
    end
end

function sanitize_for_toml(x)
    if isa(x, Symbol)
        return string(x)
    elseif isa(x, Dict)
        result = Dict{String,Any}()
        for (k, v) in x
            result[string(k)] = sanitize_for_toml(v)
        end
        return result
    elseif isa(x, Vector)
        return [sanitize_for_toml(v) for v in x]
    else
        return x
    end
end

# ----------------------
# Unified training entry
# ----------------------
"""
    train!(; modeltype::Symbol=:bnn, cfg::Dict=load_config())

Trains either a full Bayesian Neural ODE (`:bnn`) or a UDE (`:ude`), using
shared data loading and solver settings from `config/config.toml`.
Saves checkpoints in `checkpoints/` and returns a Dict of results.
"""
function train!(; modeltype::Symbol=:bnn, cfg::Dict=load_config())
    # Seeds
    base_seed = get(get(cfg, "train", Dict{String,Any}()), "seed", 42)
    Random.seed!(Int(base_seed))
    
    println("🔍 Reproducibility Info:")
    println("  → Seed: $(base_seed)")
    println("  → Julia version: $(VERSION)")
    println("  → Git SHA: $(readchomp(`git rev-parse HEAD`))")
    println("  → Timestamp: $(Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")

    # Data
    df_train = CSV.read(joinpath(@__DIR__, "..", "data", "training_dataset.csv"), DataFrame)
    subset_size = Int(getcfg(cfg, 1500, :train, :subset_size))
    df_train_subset = df_train[1:min(subset_size, nrow(df_train)), :]
    t_train = Array(df_train_subset.time)
    Y_train = Matrix(df_train_subset[:, [:x1, :x2]])
    u0_train = Y_train[1, :]

    # Solver config
    abstol = getcfg(cfg, 1e-8, :solver, :abstol)
    reltol = getcfg(cfg, 1e-8, :solver, :reltol)
    solver = Tsit5()
    solver_name = "Tsit5"
    adjoint = "none"  # training uses direct solves inside Turing likelihood here

    # MCMC config
    nsamples = Int(getcfg(cfg, 1000, :train, :samples))
    nwarmup  = Int(getcfg(cfg, 200,  :train, :warmup))
    advi_iters = Int(getcfg(cfg, 0, :train, :advi_iters))

    # Architecture
    arch_choice = String(getcfg(cfg, "baseline", :model, :arch))
    arch_sym, deriv_fn, num_params = pick_arch(arch_choice)

    meta = capture_metadata(cfg)

    if modeltype == :bnn
        @model function bayesian_neural_ode(t, Y, u0)
            # Non-centered parameterization for better HMC geometry
            σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
            θ_raw ~ MvNormal(zeros(num_params), I(num_params))  # Standard normal
            θ = 0.1 * θ_raw  # Scale to actual parameter range
            
            prob = ODEProblem(deriv_fn, u0, (minimum(t), maximum(t)), θ)
            sol = solve(prob, solver; saveat=t, abstol=abstol, reltol=reltol, maxiters=10000)
            if sol.retcode != :Success || length(sol) != length(t)
                Turing.@addlogprob! -Inf
                return
            end
            Ŷ = hcat(sol.u...)'
            for i in 1:length(t)
                Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
            end
        end
        model = bayesian_neural_ode(t_train, Y_train, u0_train)

        initial_params = (σ = 0.1, θ_raw = initial_theta("normal", 1.0, num_params))  # Unit scale for raw params
        if advi_iters > 0
            try
                q = Turing.Variational.vi(model, Turing.Variational.ADVI(advi_iters))
                if hasproperty(q, :posterior) && hasproperty(q.posterior, :μ)
                    μ = q.posterior.μ
                    if length(μ) == num_params + 1
                        initial_params = (σ = max(0.05, abs(μ[end])), θ_raw = μ[1:num_params])
                    end
                end
            catch e
                @warn "ADVI warm-start failed" error=e
            end
        end

        # Improved NUTS settings
        target_accept_val = getcfg(cfg, 0.95, :tuning, :nuts_target)  # Increased from 0.85
        target_accept = isa(target_accept_val, Vector) ? Float64(first(target_accept_val)) : Float64(target_accept_val)
        max_depth = Int(getcfg(cfg, 10, :tuning, :nuts_max_depth))
        chain = sample(model, NUTS(target_accept; max_depth=max_depth), nsamples;
                       discard_initial=nwarmup, progress=true, initial_params=initial_params)

        # Extract and transform samples back to original scale
        θ_raw_samples = Array(chain)[:, 1:num_params]
        θ_samples = 0.1 * θ_raw_samples  # Transform back to parameter scale
        σ_samples = Array(chain)[:, num_params+1]
        keep = min(500, size(θ_samples, 1))
        res = Dict(
            :params_mean => mean(θ_samples, dims=1)[1, :],
            :params_std  => std(θ_samples,  dims=1)[1, :],
            :noise_mean  => mean(σ_samples),
            :noise_std   => std(σ_samples),
            :n_samples   => size(θ_samples, 1),
            :model_type  => "bayesian_neural_ode",
            :arch        => String(arch_sym),
            :solver      => Dict(:name=>solver_name, :abstol=>abstol, :reltol=>reltol, :adjoint=>adjoint),
            :param_samples => θ_samples[1:keep, :],
            :noise_samples => σ_samples[1:keep],
            :metadata    => meta,
        )
        BSON.@save joinpath(@__DIR__, "..", "checkpoints", "bayesian_neural_ode_results.bson") bayesian_results=res
        return res

    elseif modeltype == :ude
        function ude_dynamics!(dx, x, p, t)
            x1, x2 = x
            ηin, ηout, α, β, γ = p[1:5]
            nn_params = p[6:end]
            u = Microgrid.control_input(t)
            Pgen = Microgrid.generation(t)
            Pload = Microgrid.load(t)
            Pin = u > 0 ? ηin * u : (1 / ηout) * u
            dx[1] = Pin - Microgrid.demand(t)
            nn_output = NeuralNODEArchitectures.ude_nn_forward(x1, x2, Pgen, Pload, t, nn_params)
            dx[2] = -α * x2 + nn_output + γ * x1
        end

        @model function bayesian_ude(t, Y, u0)
            # Non-centered parameterization for better HMC geometry
            σ ~ truncated(Normal(0.1, 0.05), 0.01, 0.5)
            
            # Physics parameters with non-centered parameterization
            ηin_raw ~ Normal(0, 1)
            ηin = 0.9 + 0.1 * ηin_raw
            ηout_raw ~ Normal(0, 1)
            ηout = 0.9 + 0.1 * ηout_raw
            α_raw ~ Normal(0, 1)
            α = 0.001 + 0.0005 * α_raw
            β_raw ~ Normal(0, 1)
            β = 1.0 + 0.2 * β_raw
            γ_raw ~ Normal(0, 1)
            γ = 0.001 + 0.0005 * γ_raw
            
            # Neural network parameters
            nn_params_raw ~ MvNormal(zeros(15), I(15))
            nn_params = 0.05 * nn_params_raw
            p = [ηin, ηout, α, β, γ, nn_params...]
            prob = ODEProblem(ude_dynamics!, u0, (minimum(t), maximum(t)), p)
            sol = solve(prob, solver; saveat=t, abstol=abstol, reltol=reltol, maxiters=10000)
            if sol.retcode != :Success || length(sol) != length(t)
                Turing.@addlogprob! -Inf
                return
            end
            Ŷ = hcat(sol.u...)'
            for i in 1:length(t)
                Y[i, :] ~ MvNormal(Ŷ[i, :], σ^2 * I(2))
            end
        end
        model = bayesian_ude(t_train, Y_train, u0_train)

        initial_params = (σ = 0.1, ηin_raw = 0.0, ηout_raw = 0.0, α_raw = 0.0, β_raw = 0.0, γ_raw = 0.0, nn_params_raw = initial_theta("normal", 1.0, 15))
        if advi_iters > 0
            try
                q = Turing.Variational.vi(model, Turing.Variational.ADVI(advi_iters))
                if hasproperty(q, :posterior) && hasproperty(q.posterior, :μ)
                    μ = q.posterior.μ
                    # basic guard on lengths
                    if length(μ) >= 21
                        nnμ = μ[end-14:end]
                        initial_params = (σ = max(0.05, abs(μ[1])), ηin = μ[2], ηout = μ[3], α = μ[4], β = μ[5], γ = μ[6], nn_params = nnμ)
                    end
                end
            catch e
                @warn "ADVI warm-start for UDE failed" error=e
            end
        end

        # Improved NUTS settings
        target_accept_val = getcfg(cfg, 0.95, :tuning, :nuts_target)  # Increased from 0.85
        target_accept = isa(target_accept_val, Vector) ? Float64(first(target_accept_val)) : Float64(target_accept_val)
        max_depth = Int(getcfg(cfg, 10, :tuning, :nuts_max_depth))
        chain = sample(model, NUTS(target_accept; max_depth=max_depth), nsamples;
                       discard_initial=nwarmup, progress=true, initial_params=initial_params)

        arr = Array(chain)
        # Transform raw parameters back to original scales
        ηin_vals = 0.9 .+ 0.1 .* arr[:, 2]
        ηout_vals = 0.9 .+ 0.1 .* arr[:, 3]
        α_vals = 0.001 .+ 0.0005 .* arr[:, 4]
        β_vals = 1.0 .+ 0.2 .* arr[:, 5]
        γ_vals = 0.001 .+ 0.0005 .* arr[:, 6]
        physics = hcat(ηin_vals, ηout_vals, α_vals, β_vals, γ_vals)
        neural = 0.05 .* arr[:, 7:21]  # Transform neural params back to original scale
        σs = arr[:, 1]
        keep = min(300, size(arr, 1))
        res = Dict(
            :physics_params_mean => mean(physics, dims=1)[1, :],
            :physics_params_std  => std(physics,  dims=1)[1, :],
            :neural_params_mean  => mean(neural,  dims=1)[1, :],
            :neural_params_std   => std(neural,   dims=1)[1, :],
            :noise_mean          => mean(σs),
            :noise_std           => std(σs),
            :n_samples           => size(arr, 1),
            :model_type          => "universal_differential_equation",
            :solver              => Dict(:name=>solver_name, :abstol=>abstol, :reltol=>reltol, :adjoint=>adjoint),
            :physics_samples     => physics[1:keep, :],
            :neural_samples      => neural[1:keep, :],
            :noise_samples       => σs[1:keep],
            :metadata            => meta,
        )
        BSON.@save joinpath(@__DIR__, "..", "checkpoints", "ude_results_fixed.bson") ude_results=res
        return res
    else
        error("Unknown modeltype=$(modeltype). Use :bnn or :ude")
    end
end

end # module 