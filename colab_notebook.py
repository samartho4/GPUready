# @title 🚀 Microgrid Bayesian Neural ODE Control - Enhanced Pipeline
# @markdown ## NeurIPS Submission: UDE vs BNode Comparison
# @markdown **Enhanced Pipeline with 2,880 Hyperparameter Configurations**

# @markdown ---
# @markdown ### Setup Instructions:
# @markdown 1. Run this cell to install Julia and dependencies
# @markdown 2. Upload your project files in the next cell
# @markdown 3. Run the enhanced pipeline
# @markdown 4. Download results

# Install Julia and required packages
print("🔧 Installing Julia environment...")

!wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
!tar -xzf julia-1.9.4-linux-x86_64.tar.gz
!ln -s julia-1.9.4/bin/julia /usr/local/bin/julia

# Install Julia packages
print("📦 Installing Julia packages...")
!julia -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "DifferentialEquations", "Optim", "BSON", "Statistics", "Random", "LinearAlgebra", "Dates"])'

print("✅ Julia environment ready!")
print("🚀 Ready to run your enhanced pipeline!")

# @title 📁 Upload Project Files
# @markdown Upload your project files here. You'll need:
# @markdown - data/training_roadmap.csv
# @markdown - data/validation_roadmap.csv  
# @markdown - data/test_roadmap.csv
# @markdown - scripts/enhanced_ude_tuning.jl
# @markdown - scripts/bnode_train_calibrate.jl
# @markdown - scripts/comprehensive_model_comparison.jl
# @markdown - scripts/run_enhanced_pipeline.jl

from google.colab import files
import zipfile
import os

# Create project structure
!mkdir -p microgrid-bayesian-neural-ode-control/{data,scripts,results,checkpoints}

print("📤 Please upload your project files:")
print("1. data/training_roadmap.csv")
print("2. data/validation_roadmap.csv") 
print("3. data/test_roadmap.csv")
print("4. scripts/enhanced_ude_tuning.jl")
print("5. scripts/bnode_train_calibrate.jl")
print("6. scripts/comprehensive_model_comparison.jl")
print("7. scripts/run_enhanced_pipeline.jl")

# Upload files manually through Colab interface
uploaded = files.upload()

# Move uploaded files to correct locations
for filename in uploaded.keys():
    if 'training_roadmap.csv' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/data/training_roadmap.csv
    elif 'validation_roadmap.csv' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/data/validation_roadmap.csv
    elif 'test_roadmap.csv' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/data/test_roadmap.csv
    elif 'enhanced_ude_tuning.jl' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/scripts/enhanced_ude_tuning.jl
    elif 'bnode_train_calibrate.jl' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/scripts/bnode_train_calibrate.jl
    elif 'comprehensive_model_comparison.jl' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/scripts/comprehensive_model_comparison.jl
    elif 'run_enhanced_pipeline.jl' in filename:
        !mv "{filename}" microgrid-bayesian-neural-ode-control/scripts/run_enhanced_pipeline.jl

print("✅ Files uploaded and organized!")

# @title 🚀 Execute Enhanced Pipeline
# @markdown This will run your complete enhanced pipeline with:
# @markdown - Enhanced UDE tuning (2,880 configurations, coarse search: 100)
# @markdown - BNode training with uncertainty quantification
# @markdown - Comprehensive model comparison
# @markdown - Symbolic extraction analysis
# @markdown 
# @markdown **Expected time: 3-4 hours**

import subprocess
import time
from datetime import datetime

print("🚀 Starting Enhanced Pipeline...")
print("📅 Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Expected time: 3-4 hours")
print("=" * 60)

# Change to project directory
!cd microgrid-bayesian-neural-ode-control

# Run the enhanced pipeline
start_time = time.time()

try:
    print("🔧 Step 1: Enhanced UDE Hyperparameter Tuning...")
    !julia --project=. scripts/enhanced_ude_tuning.jl
    
    print("🧠 Step 2: BNode Training and Calibration...")
    !julia --project=. scripts/bnode_train_calibrate.jl
    
    print("📊 Step 3: Comprehensive Model Comparison...")
    !julia --project=. scripts/comprehensive_model_comparison.jl
    
    print("📝 Step 4: Generating Research Summary...")
    !julia --project=. scripts/run_enhanced_pipeline.jl
    
    # Display results
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Show execution time
    end_time = time.time()
    duration = (end_time - start_time) / 3600  # hours
    print(f"⏱️ Total execution time: {duration:.2f} hours")
    
    # Display key results
    print("\n📊 Key Results:")
    !ls -la results/
    !ls -la checkpoints/
    
    # Show summary if available
    if os.path.exists('results/enhanced_pipeline_research_summary.md'):
        print("\n📋 Research Summary Preview:")
        !head -20 results/enhanced_pipeline_research_summary.md
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("🔄 Attempting fallback to standard pipeline...")
    !julia --project=. scripts/run_complete_pipeline.jl

# @title 📥 Download Results
# @markdown Download your complete results including:
# @markdown - Enhanced UDE tuning results
# @markdown - BNode posterior samples
# @markdown - Comprehensive model comparison
# @markdown - Symbolic extraction analysis
# @markdown - Research summary

import zipfile
import os

# Create results archive
print("📦 Creating results archive...")
!cd microgrid-bayesian-neural-ode-control && zip -r results_archive.zip results/ checkpoints/ *.md

# Download results
print("📥 Downloading results...")
files.download('microgrid-bayesian-neural-ode-control/results_archive.zip')

print("✅ Results downloaded successfully!")
print("📁 Archive contains:")
!cd microgrid-bayesian-neural-ode-control && ls -la results/ && ls -la checkpoints/

# @title 📊 View Results Summary
# @markdown View a summary of your results

import pandas as pd

# Try to load and display results
try:
    if os.path.exists('microgrid-bayesian-neural-ode-control/results/enhanced_ude_tuning_results.csv'):
        print("📈 Enhanced UDE Tuning Results:")
        df = pd.read_csv('microgrid-bayesian-neural-ode-control/results/enhanced_ude_tuning_results.csv')
        print(f"Total configurations tested: {len(df)}")
        print(f"Best RMSE x2: {df['mean_rmse_x2'].min():.4f}")
        print(f"Best R² x2: {df['mean_r2_x2'].max():.4f}")
        print("\nTop 5 configurations:")
        print(df.nsmallest(5, 'mean_rmse_x2')[['width', 'lambda', 'lr', 'reltol', 'mean_rmse_x2', 'mean_r2_x2']])
    
    if os.path.exists('microgrid-bayesian-neural-ode-control/results/enhanced_pipeline_research_summary.md'):
        print("\n📋 Research Summary:")
        with open('microgrid-bayesian-neural-ode-control/results/enhanced_pipeline_research_summary.md', 'r') as f:
            content = f.read()
            print(content[:1000] + "..." if len(content) > 1000 else content)
            
except Exception as e:
    print(f"❌ Error loading results: {e}")

# @title 🎯 NeurIPS Submission Status
# @markdown Check your project's readiness for NeurIPS submission

print("🎯 NeurIPS Submission Status Check")
print("=" * 50)

# Check required components
components = [
    ("Enhanced UDE Tuning", "microgrid-bayesian-neural-ode-control/results/enhanced_ude_tuning_results.csv"),
    ("BNode Training", "microgrid-bayesian-neural-ode-control/checkpoints/bnode_posterior.bson"),
    ("Model Comparison", "microgrid-bayesian-neural-ode-control/results/comprehensive_model_comparison_summary.md"),
    ("Symbolic Extraction", "microgrid-bayesian-neural-ode-control/results/symbolic_extraction_analysis.md"),
    ("Research Summary", "microgrid-bayesian-neural-ode-control/results/enhanced_pipeline_research_summary.md")
]

status = "✅ READY"
for name, path in components:
    if os.path.exists(path):
        print(f"✅ {name}: Complete")
    else:
        print(f"❌ {name}: Missing")
        status = "⚠️ INCOMPLETE"

print(f"\n🎯 Overall Status: {status}")

if status == "✅ READY":
    print("🚀 Your project is ready for NeurIPS submission!")
    print("📝 Next steps:")
    print("   1. Review results in downloaded archive")
    print("   2. Begin paper writing")
    print("   3. Create visualizations")
    print("   4. Prepare supplementary materials")
else:
    print("🔄 Some components are missing. Check the pipeline execution.")

# @title 💡 Pro Tips for Colab
# @markdown Tips to get the best results from your Colab session

print("💡 Pro Tips for Google Colab")
print("=" * 40)

tips = [
    "🔋 Use Colab Pro ($10/month) for 24+ hour runtime",
    "⚡ Enable GPU acceleration in Runtime settings",
    "💾 Download results regularly to avoid losing them",
    "📊 Monitor GPU utilization in Runtime info",
    "🔄 Save checkpoints to Colab's persistent storage",
    "📱 Use mobile hotspot if your connection is unstable",
    "⏰ Set reminders to check progress every hour",
    "📁 Keep backup of your project files"
]

for tip in tips:
    print(tip)

print("\n🎯 Your enhanced pipeline is optimized for Colab!")
print("🚀 Good luck with your NeurIPS submission!")
