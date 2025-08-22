module UncertaintyCalibration

export coverage_test, nll_score, crps_score, pit_values, calibration_plot_data

using Statistics, Distributions, LinearAlgebra

"""
    coverage_test(y_true, predictions, prediction_vars, levels=[0.5, 0.9])

Test predictive interval coverage. Returns empirical coverage for each level.

# Arguments
- `y_true`: True values (n_obs × n_dims)
- `predictions`: Predictive means (n_obs × n_dims) 
- `prediction_vars`: Predictive variances (n_obs × n_dims)
- `levels`: Coverage levels to test

# Returns
- Dictionary with empirical coverage for each level
"""
function coverage_test(y_true::AbstractMatrix, predictions::AbstractMatrix, 
                      prediction_vars::AbstractMatrix, levels::Vector{Float64}=[0.5, 0.9])
    n_obs, n_dims = size(y_true)
    coverage_results = Dict{Float64, Vector{Float64}}()
    
    for level in levels
        alpha = 1.0 - level
        z_score = quantile(Normal(), 1.0 - alpha/2.0)
        
        # Compute intervals for each dimension
        coverage_per_dim = Float64[]
        
        for dim in 1:n_dims
            lower = predictions[:, dim] - z_score * sqrt.(prediction_vars[:, dim])
            upper = predictions[:, dim] + z_score * sqrt.(prediction_vars[:, dim])
            
            # Count how many true values fall within intervals
            in_interval = (y_true[:, dim] .>= lower) .& (y_true[:, dim] .<= upper)
            empirical_coverage = mean(in_interval)
            push!(coverage_per_dim, empirical_coverage)
        end
        
        coverage_results[level] = coverage_per_dim
    end
    
    return coverage_results
end

"""
    nll_score(y_true, predictions, prediction_vars)

Compute negative log-likelihood assuming Gaussian predictive distributions.

# Arguments
- `y_true`: True values (n_obs × n_dims)
- `predictions`: Predictive means (n_obs × n_dims)
- `prediction_vars`: Predictive variances (n_obs × n_dims)

# Returns
- Average negative log-likelihood per observation
"""
function nll_score(y_true::AbstractMatrix, predictions::AbstractMatrix, 
                   prediction_vars::AbstractMatrix)
    n_obs, n_dims = size(y_true)
    total_nll = 0.0
    
    for i in 1:n_obs
        for d in 1:n_dims
            dist = Normal(predictions[i, d], sqrt(prediction_vars[i, d]))
            total_nll -= logpdf(dist, y_true[i, d])
        end
    end
    
    return total_nll / (n_obs * n_dims)
end

"""
    crps_score(y_true, predictions, prediction_vars)

Compute Continuous Ranked Probability Score for Gaussian predictive distributions.

# Arguments
- `y_true`: True values (n_obs × n_dims)
- `predictions`: Predictive means (n_obs × n_dims) 
- `prediction_vars`: Predictive variances (n_obs × n_dims)

# Returns
- Average CRPS per observation
"""
function crps_score(y_true::AbstractMatrix, predictions::AbstractMatrix,
                   prediction_vars::AbstractMatrix)
    n_obs, n_dims = size(y_true)
    total_crps = 0.0
    
    for i in 1:n_obs
        for d in 1:n_dims
            μ = predictions[i, d]
            σ = sqrt(prediction_vars[i, d])
            y = y_true[i, d]
            
            # CRPS for Gaussian distribution: analytical formula
            z = (y - μ) / σ
            crps_val = σ * (z * (2 * cdf(Normal(), z) - 1) + 2 * pdf(Normal(), z) - 1/√π)
            total_crps += crps_val
        end
    end
    
    return total_crps / (n_obs * n_dims)
end

"""
    pit_values(y_true, predictions, prediction_vars)

Compute Probability Integral Transform values for calibration assessment.

# Arguments
- `y_true`: True values (n_obs × n_dims)
- `predictions`: Predictive means (n_obs × n_dims)
- `prediction_vars`: Predictive variances (n_obs × n_dims)

# Returns
- PIT values (should be uniformly distributed if well-calibrated)
"""
function pit_values(y_true::AbstractMatrix, predictions::AbstractMatrix,
                   prediction_vars::AbstractMatrix)
    n_obs, n_dims = size(y_true)
    pit_vals = Float64[]
    
    for i in 1:n_obs
        for d in 1:n_dims
            dist = Normal(predictions[i, d], sqrt(prediction_vars[i, d]))
            push!(pit_vals, cdf(dist, y_true[i, d]))
        end
    end
    
    return pit_vals
end

"""
    calibration_plot_data(y_true, predictions, prediction_vars, n_bins=10)

Generate data for calibration plots (reliability diagrams).

# Arguments
- `y_true`: True values (n_obs × n_dims)
- `predictions`: Predictive means (n_obs × n_dims)
- `prediction_vars`: Predictive variances (n_obs × n_dims)
- `n_bins`: Number of bins for the calibration plot

# Returns
- Tuple of (predicted_probs, empirical_probs, bin_counts)
"""
function calibration_plot_data(y_true::AbstractMatrix, predictions::AbstractMatrix,
                              prediction_vars::AbstractMatrix, n_bins::Int=10)
    pit_vals = pit_values(y_true, predictions, prediction_vars)
    
    # Create bins
    bin_edges = range(0, 1, length=n_bins+1)
    predicted_probs = Float64[]
    empirical_probs = Float64[]
    bin_counts = Int[]
    
    for i in 1:n_bins
        # Find values in this bin
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        in_bin = (lower .<= pit_vals .< upper) .| (i == n_bins .& pit_vals .== 1.0)
        
        if sum(in_bin) > 0
            bin_center = (lower + upper) / 2
            emp_prob = mean(pit_vals[in_bin])
            
            push!(predicted_probs, bin_center)
            push!(empirical_probs, emp_prob)
            push!(bin_counts, sum(in_bin))
        end
    end
    
    return (predicted_probs, empirical_probs, bin_counts)
end

end # module 