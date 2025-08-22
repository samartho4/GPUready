module RoadmapPhysics

export roadmap_microgrid_ode!, generate_control_input, power_generation, power_load, create_roadmap_scenarios, validate_roadmap_physics

using Random, Distributions

"""
EXACT ROADMAP IMPLEMENTATION:
Microgrid ODE Model from July 2025 Roadmap

State Variables:
- x1(t): Energy stored in the battery [kWh] 
- x2(t): Net power flow through the grid [kW]

Parameters:
- ηin: Charging efficiency
- ηout: Discharging efficiency  
- α: Damping factor in the grid
- β: Gain on power mismatch (generation - load)
- γ: Coupling coefficient between storage and grid flow
"""

"""
    roadmap_microgrid_ode!(du, u, p, t)

Exact implementation of roadmap ODE system:

Energy Storage Dynamics:
dx1/dt = ηin·u(t)·1{u(t)>0} - (1/ηout)·u(t)·1{u(t)<0} - d(t)

Grid Power Flow Dynamics:  
dx2/dt = -αx2(t) + β·(Pgen(t) - Pload(t)) + γ·x1(t)
"""
function roadmap_microgrid_ode!(du, u, p, t)
    x1, x2 = u  # [Energy stored (kWh), Net power flow (kW)]
    ηin, ηout, α, β, γ = p[1:5]
    
    # Generate control input u(t) based on grid state
    control_input = generate_control_input(x1, x2, t)
    
    # Power demand from storage d(t)
    power_demand = 0.5 + 0.2 * sin(2π * t / 24)  # Daily demand pattern
    
    # Generation and load profiles
    P_gen = power_generation(t)
    P_load = power_load(t)
    
    # Energy Storage Dynamics (Equation 1)
    # dx1/dt = ηin·u(t)·1{u(t)>0} - (1/ηout)·u(t)·1{u(t)<0} - d(t)
    charging_term = ηin * control_input * (control_input > 0 ? 1.0 : 0.0)
    discharging_term = (1/ηout) * control_input * (control_input < 0 ? 1.0 : 0.0)
    
    du[1] = charging_term - discharging_term - power_demand
    
    # Grid Power Flow Dynamics (Equation 2)
    # dx2/dt = -αx2(t) + β·(Pgen(t) - Pload(t)) + γ·x1(t)
    du[2] = -α * x2 + β * (P_gen - P_load) + γ * x1
    
    # Apply physical constraints
    # Energy storage cannot go negative or exceed capacity
    if x1 <= 0.0 && du[1] < 0.0
        du[1] = 0.0  # Cannot discharge below zero
    elseif x1 >= 100.0 && du[1] > 0.0  # Assume 100 kWh max capacity
        du[1] = 0.0  # Cannot charge above capacity
    end
end

"""
    generate_control_input(x1, x2, t)

Generate control input u(t) for charging (+) or discharging (-) based on system state.
This represents the microgrid control strategy.
"""
function generate_control_input(x1::Float64, x2::Float64, t::Float64)
    # Control strategy based on:
    # 1. Energy level in storage (x1)
    # 2. Grid power flow situation (x2) 
    # 3. Time of day patterns
    
    hour = mod(t, 24.0)
    
    # Base control logic
    if x2 > 5.0  # Excess grid power → charge battery
        if x1 < 80.0  # Only if battery not full
            control = min(10.0, x2 * 0.5)  # Charge at rate proportional to excess
        else
            control = 0.0  # Battery nearly full
        end
    elseif x2 < -5.0  # Grid deficit → discharge battery
        if x1 > 10.0  # Only if battery has sufficient energy
            control = max(-15.0, x2 * 0.3)  # Discharge to help grid
        else
            control = 0.0  # Battery too low
        end
    else
        control = 0.0  # No significant grid imbalance
    end
    
    # Add time-based strategy (charge during low-demand periods)
    if 2.0 <= hour <= 6.0 && x1 < 60.0  # Early morning charging
        control += 2.0
    elseif 18.0 <= hour <= 22.0 && x1 > 20.0  # Evening discharge support
        control -= 1.0
    end
    
    # Add small random variations (realistic control noise)
    control += 0.1 * randn()
    
    return clamp(control, -20.0, 15.0)  # Physical limits
end

"""
    power_generation(t)

Realistic power generation profile Pgen(t) with daily solar pattern.
"""
function power_generation(t::Float64)
    hour = mod(t, 24.0)
    
    # Solar generation pattern (0-25 kW peak)
    if 6.0 <= hour <= 18.0
        base_solar = 25.0 * sin(π * (hour - 6.0) / 12.0)^2
    else
        base_solar = 0.2  # Minimal night generation
    end
    
    # Add weather variability  
    weather_factor = 0.8 + 0.4 * sin(2π * t / (24 * 7))  # Weekly weather pattern
    cloud_noise = 0.15 * randn()  # Random cloud effects
    
    generation = base_solar * weather_factor * (1 + cloud_noise)
    return max(0.0, generation)
end

"""
    power_load(t)

Realistic power load profile Pload(t) with residential/commercial patterns.
"""
function power_load(t::Float64)
    hour = mod(t, 24.0)
    
    # Daily load pattern (5-30 kW range)
    if 0.0 <= hour <= 6.0
        base_load = 8.0 + 4.0 * sin(π * hour / 6.0)  # Early morning
    elseif 6.0 <= hour <= 9.0  
        base_load = 15.0 + 8.0 * sin(π * (hour - 6.0) / 3.0)  # Morning peak
    elseif 9.0 <= hour <= 17.0
        base_load = 12.0 + 3.0 * sin(π * (hour - 9.0) / 8.0)  # Daytime
    elseif 17.0 <= hour <= 21.0
        base_load = 22.0 + 8.0 * sin(π * (hour - 17.0) / 4.0)  # Evening peak
    else
        base_load = 10.0 - 2.0 * sin(π * (hour - 21.0) / 3.0)  # Night
    end
    
    # Add realistic demand variations
    seasonal_factor = 1.0 + 0.2 * sin(2π * t / (24 * 365))  # Seasonal variation
    demand_noise = 0.1 * randn()  # Random demand fluctuations
    
    load = base_load * seasonal_factor * (1 + demand_noise)
    return max(3.0, load)  # Minimum base load
end

"""
    create_roadmap_scenarios()

Create scenarios that match the roadmap objectives for UDE and Bayesian Neural ODE testing.
"""
function create_roadmap_scenarios()
    scenarios = Dict{String, Any}()
    
    # Scenario 1: Standard residential microgrid (Baseline)
    scenarios["R1"] = Dict(
        "name" => "Residential Baseline",
        "params" => [0.92, 0.88, 0.25, 1.1, 0.05],  # [ηin, ηout, α, β, γ]
        "initial" => [50.0, 0.0],  # [50 kWh stored, 0 kW net flow]
        "description" => "Standard efficiency, moderate coupling"
    )
    
    # Scenario 2: High efficiency system (for UDE testing)
    scenarios["R2"] = Dict(
        "name" => "High Efficiency System",
        "params" => [0.95, 0.93, 0.15, 1.3, 0.03],
        "initial" => [70.0, 2.0],  # Start with more energy, slight grid export
        "description" => "High efficiency, strong generation response"
    )
    
    # Scenario 3: Grid-constrained system (for robustness testing)
    scenarios["R3"] = Dict(
        "name" => "Grid Constrained",
        "params" => [0.88, 0.85, 0.4, 0.9, 0.08],
        "initial" => [30.0, -1.5],  # Lower energy, grid import
        "description" => "Lower efficiency, high grid damping, strong coupling"
    )
    
    # Scenario 4: Variable coupling (for symbolic discovery)
    scenarios["R4"] = Dict(
        "name" => "Variable Coupling",
        "params" => [0.90, 0.87, 0.3, 1.2, 0.12],
        "initial" => [60.0, 0.5],
        "description" => "Strong storage-grid coupling for discovery"
    )
    
    return scenarios
end

"""
    validate_roadmap_physics(sol)

Validate solution against roadmap physics constraints.
"""
function validate_roadmap_physics(sol)
    violations = String[]
    
    # Check energy storage bounds
    x1_min, x1_max = extrema(sol[1, :])
    if x1_min < 0.0
        push!(violations, "Negative energy storage: $(x1_min) kWh")
    end
    if x1_max > 150.0  # Reasonable upper bound
        push!(violations, "Excessive energy storage: $(x1_max) kWh")
    end
    
    # Check power flow reasonableness
    x2_min, x2_max = extrema(sol[2, :])
    if abs(x2_min) > 100.0 || abs(x2_max) > 100.0
        push!(violations, "Extreme power flows: [$(x2_min), $(x2_max)] kW")
    end
    
    # Check for numerical issues
    all_values = vcat([u for u in sol.u]...)
    if any(!isfinite, all_values)
        push!(violations, "Non-finite values in solution")
    end
    
    return violations
end

end # module 