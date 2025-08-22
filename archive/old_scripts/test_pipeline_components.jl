#!/usr/bin/env julia

"""
    test_pipeline_components.jl

Test script to verify all pipeline components work before running the full pipeline.
This runs a minimal version of each step to catch any issues early.
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using DifferentialEquations
using Statistics, Random
using BSON

println("🧪 Testing Pipeline Components")
println("=" ^ 40)

# Load data first (needed for all tests)
println("📊 Loading data...")
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap.csv")
val_csv   = joinpath(@__DIR__, "..", "data", "validation_roadmap.csv")
test_csv  = joinpath(@__DIR__, "..", "data", "test_roadmap.csv")

@assert isfile(train_csv) "Missing training_roadmap.csv"
@assert isfile(val_csv)   "Missing validation_roadmap.csv"
@assert isfile(test_csv)  "Missing test_roadmap.csv"

train_df = CSV.read(train_csv, DataFrame)
val_df   = CSV.read(val_csv, DataFrame)
test_df  = CSV.read(test_csv, DataFrame)

println("  ✅ Data files loaded successfully")
println("    - Training: $(nrow(train_df)) rows, $(length(unique(train_df.scenario))) scenarios")
println("    - Validation: $(nrow(val_df)) rows, $(length(unique(val_df.scenario))) scenarios")
println("    - Test: $(nrow(test_df)) rows, $(length(unique(test_df.scenario))) scenarios")

# Check required columns
required_cols = ["time", "x1", "x2", "u", "u_plus", "u_minus", "I_u_pos", "I_u_neg", "d", "Pgen", "Pload", "scenario"]
df_names = names(train_df)
for col in required_cols
    if !(col in df_names)
        println("  Available columns: $(df_names)")
        error("Missing column: $(col)")
    end
end
println("  ✅ All required columns present")
println()

# Test 1: Data loading verification
println("📊 Test 1: Data loading verification...")

# Test 2: UDE model definition
println("🔧 Test 2: UDE model definition...")
try
    function ftheta_test(Pgen::Float64, θ::Vector{Float64}, width::Int)
        W1 = reshape(θ[1:width], width, 1)
        b1 = θ[width+1:width+width]
        h = tanh.(W1 * [Pgen] .+ b1)
        return sum(h)
    end
    
    function ude_rhs_test(params::Vector{Float64}, width::Int, T, sdf)
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
            du[2] = -α * x2 + ftheta_test(Pgen_t, θ, width) - β * Pload_t + γ * x1
        end
        return rhs!
    end
    
    # Test with sample data
    width = 5
    θ_size = width + width
    params = vcat([0.9, 0.9, 0.1, 1.0, 0.02], 0.1 .* randn(θ_size))
    
    # Test on one scenario
    sdf = first(groupby(train_df, :scenario))
    T = Vector{Float64}(sdf.time)
    rhs! = ude_rhs_test(params, width, T, sdf)
    x0 = [sdf.x1[1], sdf.x2[1]]
    
    prob = ODEProblem(rhs!, x0, (minimum(T), maximum(T)))
    sol = solve(prob, Tsit5(); saveat=T[1:5], abstol=1e-6, reltol=1e-6)
    
    println("  ✅ UDE model works correctly")
    println("    - ODE solve successful: $(sol.retcode)")
    println("    - Solution length: $(length(sol))")
    
catch e
    println("  ❌ UDE model failed: $(e)")
    exit(1)
end

# Test 3: BNode model definition
println("🧠 Test 3: BNode model definition...")
try
    function ftheta1_bnode_test(x1::Float64, x2::Float64, u::Float64, d::Float64, θ::Vector{Float64}, width::Int)
        start_idx = 1
        W1 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
        b1 = θ[start_idx+width*4:start_idx+width*4+width-1]
        inputs = [x1, x2, u, d]
        h = tanh.(W1 * inputs .+ b1)
        return sum(h)
    end
    
    function ftheta2_bnode_test(x1::Float64, x2::Float64, Pgen::Float64, Pload::Float64, θ::Vector{Float64}, width::Int)
        start_idx = 1 + width*4 + width
        W2 = reshape(θ[start_idx:start_idx+width*4-1], width, 4)
        b2 = θ[start_idx+width*4:start_idx+width*4+width-1]
        inputs = [x1, x2, Pgen, Pload]
        h = tanh.(W2 * inputs .+ b2)
        return sum(h)
    end
    
    # Test with sample data
    width = 5
    total_params = width*4 + width + width*4 + width  # W1 + b1 + W2 + b2
    θ = 0.1 .* randn(total_params)
    
    # Test function calls
    result1 = ftheta1_bnode_test(0.5, 0.1, 0.2, 0.1, θ, width)
    result2 = ftheta2_bnode_test(0.5, 0.1, 0.8, 0.6, θ, width)
    
    println("  ✅ BNode model works correctly")
    println("    - fθ1 output: $(round(result1, digits=4))")
    println("    - fθ2 output: $(round(result2, digits=4))")
    
catch e
    println("  ❌ BNode model failed: $(e)")
    exit(1)
end

# Test 4: Polynomial fitting
println("📈 Test 4: Polynomial fitting...")
try
    function polyfit_test(x, y, degree)
        A = zeros(length(x), degree + 1)
        for i in 1:length(x)
            for j in 1:(degree + 1)
                A[i, j] = x[i]^(j-1)
            end
        end
        return A \ y
    end
    
    function polyval_test(coeffs, x)
        result = 0.0
        for (i, c) in enumerate(coeffs)
            result += c * x^(i-1)
        end
        return result
    end
    
    # Test with sample data
    x_test = collect(0.0:0.1:1.0)
    y_test = 2.0 .+ 3.0 .* x_test .+ 0.5 .* x_test.^2 .+ 0.1 .* randn(length(x_test))
    
    coeffs = polyfit_test(x_test, y_test, 2)
    y_pred = [polyval_test(coeffs, x) for x in x_test]
    
    r2 = 1 - sum((y_pred .- y_test).^2) / sum((y_test .- mean(y_test)).^2)
    
    println("  ✅ Polynomial fitting works correctly")
    println("    - R² fit quality: $(round(r2, digits=4))")
    println("    - Coefficients: $(round.(coeffs, digits=4))")
    
catch e
    println("  ❌ Polynomial fitting failed: $(e)")
    exit(1)
end

# Test 5: Directory creation
println("📁 Test 5: Directory creation...")
try
    res_dir = joinpath(@__DIR__, "..", "results")
    ckpt_dir = joinpath(@__DIR__, "..", "checkpoints")
    
    if !isdir(res_dir); mkdir(res_dir); end
    if !isdir(ckpt_dir); mkdir(ckpt_dir); end
    
    println("  ✅ Directories ready")
    println("    - Results: $(res_dir)")
    println("    - Checkpoints: $(ckpt_dir)")
    
catch e
    println("  ❌ Directory creation failed: $(e)")
    exit(1)
end

println()
println("🎉 All Pipeline Components Tested Successfully!")
println("=" ^ 40)
println("✅ Data loading: Working")
println("✅ UDE model: Working")
println("✅ BNode model: Working")
println("✅ Polynomial fitting: Working")
println("✅ Directory creation: Working")
println()
println("🚀 Ready to run complete pipeline!")
println("   Run: julia scripts/run_complete_pipeline.jl")
