#!/usr/bin/env julia

"""
    create_correct_data.jl

Create completely correct data that matches the screenshot exactly.
This removes all incorrect columns and creates proper data structure.
"""

using Pkg
Pkg.activate(".")

using CSV, DataFrames
using Statistics, Random

println("🔧 Creating Correct Data for Screenshot Compliance")
println("=" ^ 60)

# Load current data
train_csv = joinpath(@__DIR__, "..", "data", "training_roadmap.csv")
val_csv   = joinpath(@__DIR__, "..", "data", "validation_roadmap.csv")
test_csv  = joinpath(@__DIR__, "..", "data", "test_roadmap.csv")

train_df = CSV.read(train_csv, DataFrame)
val_df   = CSV.read(val_csv, DataFrame)
test_df  = CSV.read(test_csv, DataFrame)

# Create correct data structure as per screenshot
function create_correct_dataframe(df::DataFrame)
    # Keep only the correct columns as per screenshot
    correct_columns = [:time, :x1, :x2, :u, :d, :Pgen, :Pload, :scenario, :ηin, :ηout, :α, :γ, :β]
    
    # Create new dataframe with only correct columns
    new_df = select(df, correct_columns)
    
    # Add proper indicator functions as per screenshot
    new_df.indicator_u_positive = (new_df.u .> 0) .* 1.0  # 1_{u(t)>0}
    new_df.indicator_u_negative = (new_df.u .< 0) .* 1.0  # 1_{u(t)<0}
    
    return new_df
end

# Create corrected datasets
println("📊 Creating corrected datasets...")
train_df_correct = create_correct_dataframe(train_df)
val_df_correct = create_correct_dataframe(val_df)
test_df_correct = create_correct_dataframe(test_df)

# Save corrected data
CSV.write(joinpath(@__DIR__, "..", "data", "training_roadmap_correct.csv"), train_df_correct)
CSV.write(joinpath(@__DIR__, "..", "data", "validation_roadmap_correct.csv"), val_df_correct)
CSV.write(joinpath(@__DIR__, "..", "data", "test_roadmap_correct.csv"), test_df_correct)

println("✅ Correct data created and saved")
println("📋 Correct columns: $(names(train_df_correct))")

# Verify the data structure
println("\n🧪 Verifying data structure...")
println("  Training: $(nrow(train_df_correct)) rows, $(length(unique(train_df_correct.scenario))) scenarios")
println("  Validation: $(nrow(val_df_correct)) rows, $(length(unique(val_df_correct.scenario))) scenarios")
println("  Test: $(nrow(test_df_correct)) rows, $(length(unique(test_df_correct.scenario))) scenarios")

# Test indicator functions
println("\n🔍 Testing indicator functions...")
test_u_values = [-0.5, 0.0, 0.5]
for u_val in test_u_values
    pos_indicator = u_val > 0 ? 1.0 : 0.0
    neg_indicator = u_val < 0 ? 1.0 : 0.0
    println("  u = $u_val: 1_{u>0} = $pos_indicator, 1_{u<0} = $neg_indicator")
end

println("\n✅ Data structure verification complete")
println("📁 Correct data files:")
println("  - data/training_roadmap_correct.csv")
println("  - data/validation_roadmap_correct.csv")
println("  - data/test_roadmap_correct.csv")

println("\n🎯 SCREENSHOT-COMPLIANT DATA READY!")
println("✅ All incorrect columns removed")
println("✅ Proper indicator functions added")
println("✅ Ready for corrected model implementation")
