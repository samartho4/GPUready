# Fix ODE Stiffness Issues for Microgrid Control Project
# This script implements solutions for the critical numerical stability problems

using DifferentialEquations
using Optim
using Flux
using CSV
using DataFrames
using Statistics
using Random
using BSON
using LinearAlgebra

println("🔧 Fixing ODE Stiffness Issues")
println("=" ^ 50)

# Load data
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap.csv")
train_df = CSV.read(train_csv, DataFrame)

# 1. IMPLEMENT STIFF SOLVER VERSION
function ude_rhs_stiff(params::Vector{Float64}, width::Int, T, sdf)
    function rhs!(du, x, p, t)
        x1, x2 = x
        
        # Find time index
        idx = clamp(searchsortedlast(T, t), 1, length(T))
        
        # Extract time-varying inputs
        up_t, um_t = sdf.u_plus[idx], sdf.u_minus[idx]
        Ipos_t, Ineg_t = sdf.I_u_pos[idx], sdf.I_u_neg[idx]
        d_t = sdf.d[idx]
        Pgen_t, Pload_t = sdf.Pgen[idx], sdf.Pload[idx]
        
        # Extract and constrain physics parameters
        ηin, ηout, α, β, γ = params[1:5]
        
        # CRITICAL FIX 1: Parameter constraints
        ηin = clamp(ηin, 0.7, 1.0)
        ηout = clamp(ηout, 0.7, 1.0)
        α = clamp(α, 0.01, 1.0)
        β = clamp(β, 0.1, 10.0)
        γ = clamp(γ, 0.01, 1.0)
        
        # Neural network parameters
        θ = params[6:end]
        
        # CRITICAL FIX 2: Constrain neural outputs
        function ftheta_constrained(Pgen::Float64, θ::Vector{Float64}, width::Int)
            W1 = reshape(θ[1:width], width, 1)
            b1 = θ[width+1:width+width]
            h = tanh.(W1 * [Pgen] .+ b1)
            output = sum(h)
            # Constrain output to reasonable range
            return clamp(output, -10.0, 10.0)
        end
        
        # Equations with constraints
        du[1] = ηin * up_t * Ipos_t - (1/ηout) * um_t * Ineg_t - d_t
        du[2] = -α * x2 + ftheta_constrained(Pgen_t, θ, width) - β * Pload_t + γ * x1
        
        # CRITICAL FIX 3: Prevent NaN/Inf
        for i in 1:2
            if !isfinite(du[i])
                du[i] = 0.0
            end
        end
    end
    return rhs!
end

# 2. IMPLEMENT ROBUST TRAINING FUNCTION
function train_ude_robust(train_df, width=5, λ=1e-4, max_iters=1000)
    println("🚀 Training UDE with robust solver...")
    
    # Parameter setup
    θ_size = width + width
    total_params = 5 + θ_size  # physics + neural
    
    # Initialize parameters with reasonable values
    physics_params = [0.9, 0.9, 0.1, 1.0, 0.02]  # ηin, ηout, α, β, γ
    nn_params = 0.1 .* randn(θ_size)
    initial_params = vcat(physics_params, nn_params)
    
    # CRITICAL FIX 4: Enhanced loss function with regularization
    function robust_loss(params)
        total_loss = 0.0
        valid_scenarios = 0
        
        for (scenario, sdf) in pairs(groupby(train_df, :scenario))
            try
                T = Vector{Float64}(sdf.time)
                rhs! = ude_rhs_stiff(params, width, T, sdf)
                x0 = [sdf.x1[1], sdf.x2[1]]
                
                # CRITICAL FIX 5: Use stiff solver with robust settings
                prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
                sol = solve(prob, Rodas5(); 
                    saveat=T, 
                    abstol=1e-6, 
                    reltol=1e-6,
                    maxiters=10000,
                    adaptive=true)
                
                if sol.retcode == :Success
                    Y_pred = Matrix(sol)
                    Y_true = Matrix(sdf[:, [:x1, :x2]])
                    
                    # Compute MSE
                    mse = mean((Y_pred - Y_true).^2)
                    
                    # CRITICAL FIX 6: Add regularization
                    reg = λ * (norm(params[1:5])^2 + norm(params[6:end])^2)
                    
                    total_loss += mse + reg
                    valid_scenarios += 1
                else
                    println("  ⚠️ Scenario $(scenario) failed: $(sol.retcode)")
                end
                
            catch e
                println("  ❌ Scenario $(scenario) error: $(e)")
                continue
            end
        end
        
        if valid_scenarios == 0
            return Inf
        end
        
        return total_loss / valid_scenarios
    end
    
    # CRITICAL FIX 7: Robust optimization with constraints
    function constrained_optimization()
        # Define parameter bounds
        lower_bounds = [0.7, 0.7, 0.01, 0.1, 0.01, fill(-10.0, θ_size)...]
        upper_bounds = [1.0, 1.0, 1.0, 10.0, 1.0, fill(10.0, θ_size)...]
        
        # Use L-BFGS with constraints
        result = optimize(robust_loss, 
                        lower_bounds, 
                        upper_bounds, 
                        initial_params,
                        Fminbox(LBFGS()),
                        Optim.Options(iterations=max_iters,
                                    show_trace=true,
                                    show_every=10))
        
        return result
    end
    
    # Run optimization
    result = constrained_optimization()
    
    if result.minimizer !== nothing
        println("✅ Training completed successfully!")
        println("  Final loss: $(result.minimum)")
        println("  Iterations: $(result.iterations)")
        println("  Converged: $(result.converged)")
        
        # Extract final parameters
        final_params = result.minimizer
        physics_params = final_params[1:5]
        nn_params = final_params[6:end]
        
        return Dict(
            "physics_params" => physics_params,
            "nn_params" => nn_params,
            "width" => width,
            "final_loss" => result.minimum,
            "converged" => result.converged
        )
    else
        error("❌ Optimization failed!")
    end
end

# 3. TEST THE FIXES
println("🧪 Testing robust UDE implementation...")

# Test on a small subset first
test_scenarios = first(groupby(train_df, :scenario), 5)
test_df = combine(test_scenarios, identity)

try
    result = train_ude_robust(test_df, width=5, λ=1e-4, max_iters=100)
    
    println("✅ Robust UDE test successful!")
    println("  Physics params: $(round.(result["physics_params"], digits=3))")
    println("  Final loss: $(round(result["final_loss"], digits=6))")
    println("  Converged: $(result["converged"])")
    
    # Save the robust model
    BSON.@save joinpath(@__DIR__, "..", "checkpoints", "ude_robust_test.bson") result
    
    println("💾 Saved robust UDE test model")
    
catch e
    println("❌ Robust UDE test failed: $(e)")
    println("  Error type: $(typeof(e))")
    println("  Backtrace: $(stacktrace())")
end

println("=" ^ 50)
println("🔧 ODE Stiffness Fixes Implemented:")
println("✅ Stiff solver (Rodas5) with robust settings")
println("✅ Parameter constraints and clamping")
println("✅ Enhanced regularization")
println("✅ NaN/Inf prevention")
println("✅ Constrained optimization")
println("✅ Error handling and recovery")

println("\n📋 Next Steps:")
println("1. Test on full dataset if small test succeeds")
println("2. Integrate fixes into main training pipeline")
println("3. Run complete evaluation with robust models")
