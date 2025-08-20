#!/usr/bin/env julia

"""
    run_complete_pipeline.jl

Master orchestrator:
  1) Tune UDE hyperparameters → checkpoints/ude_best_tuned.bson
  2) Train and calibrate BNode on subset → checkpoints/bnode_posterior.bson
  3) Compare models and extract UDE symbolic form → results/*
"""

using Pkg
Pkg.activate(".")

using Dates

function run_cmd(cmd::Cmd)
    println("▶️  Running: ", cmd)
    run(cmd)
end

project_root = normpath(joinpath(@__DIR__, ".."))

# 1) UDE tuning
ude_ckpt = joinpath(project_root, "checkpoints", "ude_best_tuned.bson")
if !isfile(ude_ckpt)
    log = joinpath(project_root, "results", "ude_tuning_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".log")
    cmd = `julia --project=. $(joinpath(project_root, "scripts", "tune_ude_hparams.jl")) > $(log) 2>&1`
    run_cmd(cmd)
end
@assert isfile(ude_ckpt) "UDE tuning did not produce ude_best_tuned.bson"

# 2) BNode training/calibration
bnode_ckpt = joinpath(project_root, "checkpoints", "bnode_posterior.bson")
if !isfile(bnode_ckpt)
    cmd = `julia --project=. $(joinpath(project_root, "scripts", "bnode_train_calibrate.jl"))`
    run_cmd(cmd)
end
@assert isfile(bnode_ckpt) "BNode training did not produce bnode_posterior.bson"

# 3) Comprehensive comparison
cmd = `julia --project=. $(joinpath(project_root, "scripts", "comprehensive_model_comparison.jl"))`
run_cmd(cmd)

println("✅ Pipeline complete. See checkpoints/ and results/ for outputs.")


