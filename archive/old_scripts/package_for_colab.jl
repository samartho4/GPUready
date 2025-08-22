#!/usr/bin/env julia

"""
    package_for_colab.jl

Package essential project files for Google Colab deployment.
Creates a zip file with all necessary files for running the enhanced pipeline.
"""

using Pkg
Pkg.activate(".")

println("📦 Packaging Project for Google Colab")
println("=" ^ 50)

# Create temporary directory for packaging
package_dir = "colab_package"
if isdir(package_dir)
    rm(package_dir, recursive=true)
end
mkdir(package_dir)

# Create directory structure
mkdir(joinpath(package_dir, "data"))
mkdir(joinpath(package_dir, "scripts"))
mkdir(joinpath(package_dir, "results"))
mkdir(joinpath(package_dir, "checkpoints"))

# Copy essential data files
println("📊 Copying data files...")
data_files = [
    "data/training_roadmap.csv",
    "data/validation_roadmap.csv", 
    "data/test_roadmap.csv"
]

for file in data_files
    if isfile(file)
        cp(file, joinpath(package_dir, file))
        println("  ✅ $file")
    else
        println("  ❌ Missing: $file")
    end
end

# Copy essential scripts
println("🔧 Copying scripts...")
script_files = [
    "scripts/enhanced_ude_tuning.jl",
    "scripts/bnode_train_calibrate.jl",
    "scripts/comprehensive_model_comparison.jl",
    "scripts/run_enhanced_pipeline.jl",
    "scripts/run_complete_pipeline.jl",  # fallback
    "scripts/test_pipeline_components.jl"
]

for file in script_files
    if isfile(file)
        cp(file, joinpath(package_dir, file))
        println("  ✅ $file")
    else
        println("  ❌ Missing: $file")
    end
end

# Copy project files
println("📋 Copying project files...")
project_files = [
    "Project.toml",
    "Manifest.toml",
    "README.md",
    "PROJECT_STATUS.md"
]

for file in project_files
    if isfile(file)
        cp(file, joinpath(package_dir, file))
        println("  ✅ $file")
    else
        println("  ❌ Missing: $file")
    end
end

# Create README for Colab
colab_readme = """
# Microgrid Bayesian Neural ODE Control - Colab Package

## 🚀 Quick Start for Google Colab

### 1. Upload Files
Upload all files in this package to your Colab environment.

### 2. Install Julia
```python
!wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
!tar -xzf julia-1.9.4-linux-x86_64.tar.gz
!ln -s julia-1.9.4/bin/julia /usr/local/bin/julia
```

### 3. Install Dependencies
```python
!julia -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "DifferentialEquations", "Optim", "BSON", "Statistics", "Random", "LinearAlgebra", "Dates"])'
```

### 4. Run Enhanced Pipeline
```python
!julia --project=. scripts/run_enhanced_pipeline.jl
```

## 📁 File Structure
- `data/`: Training, validation, and test datasets
- `scripts/`: Enhanced pipeline scripts
- `results/`: Output results (created during execution)
- `checkpoints/`: Model checkpoints (created during execution)

## ⏱️ Expected Runtime
- **Enhanced Pipeline**: 3-4 hours
- **Standard Pipeline**: 2-3 hours (fallback)

## 🎯 NeurIPS Submission
This package contains everything needed for your NeurIPS submission:
- Enhanced UDE hyperparameter tuning (5,760 configurations)
- BNode training with uncertainty quantification
- Comprehensive model comparison
- Symbolic extraction analysis

## 📊 Expected Results
- Enhanced UDE tuning results
- BNode posterior samples
- Model comparison metrics
- Symbolic extraction quality
- Research summary for NeurIPS

Good luck with your submission! 🚀
"""

open(joinpath(package_dir, "COLAB_README.md"), "w") do f
    write(f, colab_readme)
end

println("  ✅ COLAB_README.md")

# Create zip file
println("📦 Creating zip package...")
zip_file = "microgrid_colab_package.zip"
if isfile(zip_file)
    rm(zip_file)
end

# Use system zip command
run(`zip -r $zip_file $package_dir`)

# Clean up
rm(package_dir, recursive=true)

println("✅ Package created: $zip_file")
println("📁 Package contents:")
run(`unzip -l $zip_file`)

println("\n🚀 Ready for Google Colab!")
println("📤 Upload $zip_file to Colab and extract")
println("📋 Follow instructions in COLAB_README.md")
