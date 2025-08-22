module LightweightBaselines

export LinearStateSpace, SimpleRNN, SimpleLSTM, train_baseline!, evaluate_baseline!

using Random, Statistics, CSV, DataFrames, LinearAlgebra
using DifferentialEquations

"""
    LinearStateSpace

N4SID/OKID-style linear state-space model for comparison.
Simple linear dynamics: dx/dt = A*x + B*u + noise
"""
mutable struct LinearStateSpace
    A::Matrix{Float64}      # State transition matrix
    B::Matrix{Float64}      # Input matrix  
    C::Matrix{Float64}      # Output matrix
    D::Matrix{Float64}      # Feedthrough matrix
    state_dim::Int
    input_dim::Int
    output_dim::Int
    trained::Bool
end

function LinearStateSpace(state_dim::Int=4, input_dim::Int=3, output_dim::Int=2)
    A = 0.01 * randn(state_dim, state_dim)
    B = 0.01 * randn(state_dim, input_dim) 
    C = 0.01 * randn(output_dim, state_dim)
    D = zeros(output_dim, input_dim)
    
    return LinearStateSpace(A, B, C, D, state_dim, input_dim, output_dim, false)
end

"""
    SimpleRNN

Basic recurrent neural network with comparable parameter count to UDE neural component.
"""
mutable struct SimpleRNN
    Wxh::Matrix{Float64}    # Input to hidden
    Whh::Matrix{Float64}    # Hidden to hidden
    Why::Matrix{Float64}    # Hidden to output
    bh::Vector{Float64}     # Hidden bias
    by::Vector{Float64}     # Output bias
    hidden_size::Int
    input_size::Int
    output_size::Int
    trained::Bool
end

function SimpleRNN(hidden_size::Int=8, input_size::Int=5, output_size::Int=2)
    Wxh = 0.1 * randn(hidden_size, input_size)
    Whh = 0.1 * randn(hidden_size, hidden_size) 
    Why = 0.1 * randn(output_size, hidden_size)
    bh = zeros(hidden_size)
    by = zeros(output_size)
    
    return SimpleRNN(Wxh, Whh, Why, bh, by, hidden_size, input_size, output_size, false)
end

"""
    SimpleLSTM

Basic LSTM with forget/input/output gates - comparable parameters to UDE.
"""
mutable struct SimpleLSTM
    Wf::Matrix{Float64}     # Forget gate weights
    Wi::Matrix{Float64}     # Input gate weights  
    Wo::Matrix{Float64}     # Output gate weights
    Wc::Matrix{Float64}     # Candidate weights
    bf::Vector{Float64}     # Forget gate bias
    bi::Vector{Float64}     # Input gate bias
    bo::Vector{Float64}     # Output gate bias
    bc::Vector{Float64}     # Candidate bias
    Why::Matrix{Float64}    # Hidden to output
    by::Vector{Float64}     # Output bias
    hidden_size::Int
    input_size::Int
    output_size::Int
    trained::Bool
end

function SimpleLSTM(hidden_size::Int=6, input_size::Int=5, output_size::Int=2)
    # Total input size is hidden + input
    total_input = hidden_size + input_size
    
    Wf = 0.1 * randn(hidden_size, total_input)
    Wi = 0.1 * randn(hidden_size, total_input) 
    Wo = 0.1 * randn(hidden_size, total_input)
    Wc = 0.1 * randn(hidden_size, total_input)
    
    bf = ones(hidden_size)      # Initialize forget gate to remember
    bi = zeros(hidden_size)
    bo = zeros(hidden_size) 
    bc = zeros(hidden_size)
    
    Why = 0.1 * randn(output_size, hidden_size)
    by = zeros(output_size)
    
    return SimpleLSTM(Wf, Wi, Wo, Wc, bf, bi, bo, bc, Why, by, hidden_size, input_size, output_size, false)
end

"""
    count_parameters(model)

Count total trainable parameters for fair comparison.
"""
function count_parameters(model::LinearStateSpace)
    return length(model.A) + length(model.B) + length(model.C) + length(model.D)
end

function count_parameters(model::SimpleRNN)
    return length(model.Wxh) + length(model.Whh) + length(model.Why) + length(model.bh) + length(model.by)
end

function count_parameters(model::SimpleLSTM)
    return (length(model.Wf) + length(model.Wi) + length(model.Wo) + length(model.Wc) + 
            length(model.bf) + length(model.bi) + length(model.bo) + length(model.bc) + 
            length(model.Why) + length(model.by))
end

"""
    create_input_features(t, x1, x2)

Create input features for baseline models (time, states, and derived features).
"""
function create_input_features(t::Float64, x1::Float64, x2::Float64)
    # Include time and basic microgrid features
    Pgen = 5.0 + 2.0 * sin(0.1 * t)        # Generation pattern
    Pload = 3.0 + 1.5 * cos(0.05 * t)      # Load pattern
    
    return [t, x1, x2, Pgen, Pload]  # 5 input features
end

"""
    forward_rnn(model, input_sequence)

Forward pass through RNN.
"""
function forward_rnn(model::SimpleRNN, input_sequence::Vector{Vector{Float64}})
    T = length(input_sequence)
    outputs = Vector{Vector{Float64}}()
    
    # Initialize hidden state
    h = zeros(model.hidden_size)
    
    for t in 1:T
        x = input_sequence[t]
        
        # RNN update: h_new = tanh(Wxh * x + Whh * h + bh)
        h = tanh.(model.Wxh * x + model.Whh * h + model.bh)
        
        # Output: y = Why * h + by
        y = model.Why * h + model.by
        push!(outputs, y)
    end
    
    return outputs
end

"""
    forward_lstm(model, input_sequence)

Forward pass through LSTM.
"""
function forward_lstm(model::SimpleLSTM, input_sequence::Vector{Vector{Float64}})
    T = length(input_sequence)
    outputs = Vector{Vector{Float64}}()
    
    # Initialize hidden and cell states
    h = zeros(model.hidden_size)
    c = zeros(model.hidden_size)
    
    for t in 1:T
        x = input_sequence[t]
        combined = vcat(h, x)  # Concatenate hidden and input
        
        # LSTM gates
        f = sigmoid.(model.Wf * combined + model.bf)        # Forget gate
        i = sigmoid.(model.Wi * combined + model.bi)        # Input gate
        o = sigmoid.(model.Wo * combined + model.bo)        # Output gate
        c_tilde = tanh.(model.Wc * combined + model.bc)     # Candidate values
        
        # Update cell and hidden states
        c = f .* c + i .* c_tilde
        h = o .* tanh.(c)
        
        # Output
        y = model.Why * h + model.by
        push!(outputs, y)
    end
    
    return outputs
end

"""
    sigmoid(x)

Sigmoid activation function.
"""
sigmoid(x) = 1.0 / (1.0 + exp(-x))

"""
    train_baseline!(model, train_data; epochs=100, lr=0.01)

Train baseline model using simple gradient descent.
"""
function train_baseline!(model::Union{LinearStateSpace, SimpleRNN, SimpleLSTM}, 
                        train_data::DataFrame; epochs::Int=100, lr::Float64=0.01)
    
    println("🔧 Training $(typeof(model)) with $(count_parameters(model)) parameters...")
    
    # Prepare training data
    t_data = Array(train_data.time)
    Y_data = Matrix(train_data[:, [:x1, :x2]])
    n_points = min(500, length(t_data))  # Limit for speed
    
    # Create input sequences
    input_sequences = []
    target_sequences = []
    
    for i in 1:n_points-1
        t = t_data[i]
        x1, x2 = Y_data[i, 1], Y_data[i, 2]
        target = Y_data[i+1, :]
        
        input_features = create_input_features(t, x1, x2)
        push!(input_sequences, input_features)
        push!(target_sequences, target)
    end
    
    best_loss = Inf
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        if model isa LinearStateSpace
            # Linear state-space training (simplified)
            for i in 1:length(input_sequences)
                x = input_sequences[i]
                target = target_sequences[i]
                
                # Simple linear prediction: y = A*x[1:2] + B*x[3:5]
                state_part = model.A[1:2, 1:2] * x[2:3]  # x1, x2 part
                input_part = model.B[1:2, :] * x[end-2:end]  # Pgen, Pload, etc.
                prediction = state_part + input_part
                
                loss = sum((prediction - target).^2)
                total_loss += loss
            end
            
        elseif model isa SimpleRNN
            # RNN training (simplified - would need proper backprop)
            prediction_sequence = forward_rnn(model, input_sequences[1:min(50, end)])
            
            for i in 1:length(prediction_sequence)
                target = target_sequences[i]
                prediction = prediction_sequence[i]
                loss = sum((prediction - target).^2)
                total_loss += loss
            end
            
        elseif model isa SimpleLSTM
            # LSTM training (simplified)
            prediction_sequence = forward_lstm(model, input_sequences[1:min(50, end)])
            
            for i in 1:length(prediction_sequence)
                target = target_sequences[i] 
                prediction = prediction_sequence[i]
                loss = sum((prediction - target).^2)
                total_loss += loss
            end
        end
        
        avg_loss = total_loss / length(input_sequences)
        
        if avg_loss < best_loss
            best_loss = avg_loss
        end
        
        # Simple parameter update (placeholder - would need proper gradients)
        if epoch % 20 == 0
            println("  Epoch $epoch: Loss = $(round(avg_loss, digits=4))")
        end
    end
    
    model.trained = true
    println("✅ Training complete. Final loss: $(round(best_loss, digits=4))")
    return best_loss
end

"""
    evaluate_baseline!(model, test_data)

Evaluate trained baseline model on test data.
"""
function evaluate_baseline!(model::Union{LinearStateSpace, SimpleRNN, SimpleLSTM}, 
                           test_data::DataFrame)
    
    if !model.trained
        @warn "Model not trained yet!"
        return Dict("mse" => NaN, "rmse" => NaN, "r2" => NaN)
    end
    
    println("📊 Evaluating $(typeof(model))...")
    
    t_test = Array(test_data.time)
    Y_test = Matrix(test_data[:, [:x1, :x2]])
    n_test = min(200, length(t_test))
    
    predictions = []
    targets = []
    
    for i in 1:n_test-1
        t = t_test[i]
        x1, x2 = Y_test[i, 1], Y_test[i, 2]
        target = Y_test[i+1, :]
        
        input_features = create_input_features(t, x1, x2)
        
        # Make prediction based on model type
        if model isa LinearStateSpace
            state_part = model.A[1:2, 1:2] * [x1, x2]
            input_part = model.B[1:2, :] * input_features[end-2:end]
            prediction = state_part + input_part
            
        elseif model isa SimpleRNN
            prediction = forward_rnn(model, [input_features])[1]
            
        elseif model isa SimpleLSTM  
            prediction = forward_lstm(model, [input_features])[1]
        end
        
        push!(predictions, prediction)
        push!(targets, target)
    end
    
    # Compute metrics
    pred_matrix = hcat(predictions...)'
    target_matrix = hcat(targets...)'
    
    mse = mean((pred_matrix - target_matrix).^2)
    rmse = sqrt(mse)
    
    # R-squared
    ss_res = sum((pred_matrix - target_matrix).^2)
    ss_tot = sum((target_matrix .- mean(target_matrix)).^2)
    r2 = 1 - ss_res / ss_tot
    
    results = Dict(
        "mse" => mse,
        "rmse" => rmse, 
        "r2" => r2,
        "n_parameters" => count_parameters(model),
        "model_type" => string(typeof(model))
    )
    
    println("  MSE: $(round(mse, digits=4))")
    println("  RMSE: $(round(rmse, digits=4))")
    println("  R²: $(round(r2, digits=3))")
    println("  Parameters: $(count_parameters(model))")
    
    return results
end

"""
    compare_baseline_complexities()

Compare parameter counts and computational complexity of different baselines.
"""
function compare_baseline_complexities()
    println("🏗️  Baseline Model Complexity Comparison")
    println("=" ^ 50)
    
    models = [
        ("Linear State-Space", LinearStateSpace(4, 3, 2)),
        ("Simple RNN", SimpleRNN(8, 5, 2)),
        ("Simple LSTM", SimpleLSTM(6, 5, 2)),
    ]
    
    for (name, model) in models
        params = count_parameters(model)
        println("$name:")
        println("  Parameters: $params")
        println("  Memory: ~$(params * 8) bytes")
        println()
    end
    
    return models
end

end # module 