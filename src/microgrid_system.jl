module Microgrid

export microgrid_ode!, generation, load, control_input, demand, create_scenarios, validate_physics

using DifferentialEquations, Random, Distributions

"""
    Microgrid State Variables:
    x1 = Battery State of Charge (SOC) [dimensionless, 0-1]  
    x2 = Power Imbalance [kW] (positive = excess, negative = deficit)
    
    Physical Parameters:
    ηin  = Battery charging efficiency [0.85-0.95]
    ηout = Battery discharging efficiency [0.85-0.95] 
    α    = Grid coupling coefficient [0.1-0.5]
    β    = Load response coefficient [0.8-1.5]
    γ    = Generation variability [0.2-0.6]
"""

"""
    microgrid_ode!(du, u, p, t)

ORIGINAL EQUATIONS FROM PAPER with SOC bounds fix:
- dx1/dt = ηin * u(t) * 1{u(t)>0} - (1/ηout) * u(t) * 1{u(t)<0} - d(t)
- dx2/dt = -α * x2(t) + β * (Pgen(t) - Pload(t)) + γ * x1(t)
- CRITICAL FIX: SOC constrained to [0.0, 1.0] range
"""
function microgrid_ode!(du, u, p, t)
    x1, x2 = u  # SOC, Power imbalance
    ηin, ηout, α, β, γ = p[1:5]
    
    # CRITICAL FIX: Ensure SOC stays in physical bounds
    x1_clamped = clamp(x1, 0.05, 0.95)
    
    # Generation and load profiles (from paper)
    P_gen = generation(t, γ)
    P_load = load(t, β)
    
    # Control input u(t) - simplified as power imbalance
    u_t = P_gen - P_load
    
    # Demand term d(t) - simplified as constant load
    d_t = P_load * 0.1  # 10% of load as storage demand
    
    # ORIGINAL EQUATION 1: Energy Storage Dynamics
    # dx1/dt = ηin * u(t) * 1{u(t)>0} - (1/ηout) * u(t) * 1{u(t)<0} - d(t)
    if u_t > 0
        du[1] = ηin * u_t - d_t  # Charging
    else
        du[1] = (1/ηout) * u_t - d_t  # Discharging
    end
    
    # ORIGINAL EQUATION 2: Grid Power Flow Dynamics  
    # dx2/dt = -α * x2(t) + β * (Pgen(t) - Pload(t)) + γ * x1(t)
    du[2] = -α * x2 + β * (P_gen - P_load) + γ * x1_clamped
    
    # CRITICAL FIX: Ensure derivatives don't cause unphysical states
    if x1_clamped <= 0.05 && du[1] < 0
        du[1] = 0.0  # Don't discharge below minimum
    elseif x1_clamped >= 0.95 && du[1] > 0
        du[1] = 0.0  # Don't charge above maximum
    end
end

"""
    generation(t, γ=0.4)

Realistic solar generation profile with variability.
"""
function generation(t, γ=0.4)
    hour = mod(t, 24.0)
    
    # Base solar profile (0-20 kW peak)
    if 6 <= hour <= 18
        base = 20.0 * sin(π * (hour - 6) / 12)^2
    else
        base = 0.1  # Minimal nighttime generation
    end
    
    # Add realistic variability (clouds, etc.)
    noise = γ * randn() * sqrt(base)
    return max(0.0, base + noise)
end

"""
    load(t, β=1.2) 

Realistic residential/commercial load profile.
"""
function load(t, β=1.2)
    hour = mod(t, 24.0)
    
    # Typical daily load pattern (5-25 kW)
    if 0 <= hour <= 6
        base = 8.0 + 3.0 * sin(π * hour / 6)  # Early morning
    elseif 6 <= hour <= 9
        base = 15.0 + 5.0 * sin(π * (hour - 6) / 3)  # Morning peak
    elseif 9 <= hour <= 17
        base = 12.0 + 2.0 * sin(π * (hour - 9) / 8)  # Daytime
    elseif 17 <= hour <= 21
        base = 20.0 + 5.0 * sin(π * (hour - 17) / 4)  # Evening peak
    else
        base = 10.0 - 2.0 * sin(π * (hour - 21) / 3)  # Night
    end
    
    # Load variability
    noise = 0.1 * β * randn() * sqrt(base)
    return max(2.0, base + noise)
end

"""
    control_input(t)

Generate control input u(t) for charging (+) or discharging (-) based on time.
This represents the microgrid control strategy.
"""
function control_input(t::Float64)
    # Control strategy based on time of day patterns
    hour = mod(t, 24.0)
    
    # Base control logic - simplified for training
    if 2.0 <= hour <= 6.0  # Early morning charging
        control = 2.0
    elseif 18.0 <= hour <= 22.0  # Evening discharge support
        control = -1.0
    else
        control = 0.0  # No significant grid imbalance
    end
    
    # Add small random variations (realistic control noise)
    control += 0.1 * randn()
    
    return clamp(control, -5.0, 5.0)  # Physical limits
end

"""
    demand(t)

Power demand from storage d(t) - simplified as time-varying load.
"""
function demand(t::Float64)
    hour = mod(t, 24.0)
    
    # Daily demand pattern (0.1-0.8 kW)
    if 0 <= hour <= 6
        base = 0.2 + 0.1 * sin(π * hour / 6)  # Early morning
    elseif 6 <= hour <= 9
        base = 0.4 + 0.2 * sin(π * (hour - 6) / 3)  # Morning peak
    elseif 9 <= hour <= 17
        base = 0.3 + 0.1 * sin(π * (hour - 9) / 8)  # Daytime
    elseif 17 <= hour <= 21
        base = 0.6 + 0.2 * sin(π * (hour - 17) / 4)  # Evening peak
    else
        base = 0.2 - 0.1 * sin(π * (hour - 21) / 3)  # Night
    end
    
    # Add small variability
    noise = 0.05 * randn() * sqrt(base)
    return max(0.05, base + noise)
end

"""
    create_scenarios()

Generate physically meaningful scenario parameters.
"""
function create_scenarios()
    scenarios = Dict{String, Any}()
    
    # Scenario 1: Standard residential (baseline)
    scenarios["S1"] = Dict(
        "name" => "Residential Baseline",
        "params" => [0.90, 0.90, 0.3, 1.2, 0.4],  # [ηin, ηout, α, β, γ]
        "initial" => [0.5, 0.0],  # [SOC=50%, no imbalance]
        "description" => "Standard residential microgrid"
    )
    
    # Scenario 2: High efficiency system
    scenarios["S2"] = Dict(
        "name" => "High Efficiency",
        "params" => [0.95, 0.93, 0.25, 1.0, 0.3],
        "initial" => [0.7, 2.0],  # Start with charged battery, slight excess
        "description" => "High-efficiency batteries, stable generation"
    )
    
    # Scenario 3: Variable generation (cloudy)
    scenarios["S3"] = Dict(
        "name" => "Variable Generation", 
        "params" => [0.88, 0.87, 0.4, 1.3, 0.6],
        "initial" => [0.3, -1.5],  # Low battery, power deficit
        "description" => "High generation variability (cloudy weather)"
    )
    
    # Scenario 4: High load variability
    scenarios["S4"] = Dict(
        "name" => "Variable Load",
        "params" => [0.91, 0.89, 0.35, 1.5, 0.35],
        "initial" => [0.6, 1.0],
        "description" => "High load variability (commercial/industrial)"
    )
    
    # Scenario 5: Grid-constrained
    scenarios["S5"] = Dict(
        "name" => "Grid Constrained",
        "params" => [0.86, 0.85, 0.5, 1.1, 0.45],
        "initial" => [0.4, -0.5],
        "description" => "Weak grid connection, high coupling"
    )
    
    return scenarios
end

"""
    validate_physics(sol)

Check if solution satisfies physical constraints.
"""
function validate_physics(sol)
    violations = String[]
    
    # Check SOC bounds (CRITICAL: Must be in [0,1])
    x1_min, x1_max = extrema(sol[1, :])
    if x1_min < 0.0 || x1_max > 1.0
        push!(violations, "SOC out of bounds: [$x1_min, $x1_max]")
    end
    
    # Check power imbalance reasonable
    x2_min, x2_max = extrema(sol[2, :])
    if abs(x2_min) > 50.0 || abs(x2_max) > 50.0
        push!(violations, "Power imbalance extreme: [$x2_min, $x2_max] kW")
    end
    
    # Check for NaN/Inf in all solution values
    all_values = vcat([u for u in sol.u]...)  # Flatten trajectory data
    if any(!isfinite, all_values)
        push!(violations, "Non-finite values in solution")
    end
    
    return violations
end

end # module 