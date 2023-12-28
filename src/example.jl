using DeepAR
using CSV
using DataFrames
using Statistics

function split_data(df::DataFrame, train_ratio::Float64)
    # Calculate the size of the training set
    train_size = floor(Int, train_ratio * size(df, 1))

    # Split the DataFrame without shuffling
    train_set = df[1:train_size, :]
    test_set = df[(train_size+1):end, :]

    return train_set, test_set
end

function impute_mean(df::DataFrame, col::Symbol)
    mean_val = mean(skipmissing(df[!, col]))
    df[!, col] = coalesce.(df[!, col], mean_val)
end

function impute_median(df::DataFrame, col::Symbol)
    median_val = median(skipmissing(df[!, col]))
    df[!, col] = coalesce.(df[!, col], median_val)
end

function impute_with_surrounding_median(df::DataFrame, col::Symbol, num_elements::Int)
    data = df[!, col]
    for i in 1:length(data)
        if ismissing(data[i])
            start_idx = max(1, i - num_elements)
            end_idx = min(length(data), i + num_elements)
            surrounding_values = data[start_idx:end_idx]
            non_missing_values = collect(skipmissing(surrounding_values))
            data[i] = median(non_missing_values)
        end
    end
    df[!, col] = data
    return df  # Return the entire DataFrame
end

function normalize_df!(df::DataFrame)
    μ = 0.0f0
    σ = 0.0f0
    for col in names(df)
        if eltype(df[!, col]) <: Union{Missing,Number}  # Check if the column is numeric
            μ = mean(skipmissing(df[!, col]))
            σ = std(skipmissing(df[!, col]))
            if σ != 0
                df[!, col] = (df[!, col] .- μ) ./ σ
            end
        end
    end
    return (df, μ, σ)
end

function normalize_column!(df::DataFrame, col::Symbol, μ::Float64, σ::Float64)
    if σ != 0
        df[!, col] = (df[!, col] .- μ) ./ σ
    end
end

function moving_average(data::Vector, window_size::Int)
    n = length(data)
    ma = Vector{Float64}(undef, n - window_size + 1)
    for i in 1:(n - window_size + 1)
        ma[i] = mean(data[i:(i + window_size - 1)])
    end
    return ma
end

function evaluate_model(model, loaderXtrain, loaderYtrain, n_samples=100)
    likelihoods = []
    for (batch_Xₜ, batch_Xₜ₊₁) in zip(loaderXtrain, loaderYtrain)
        likelihood = 0.0f0
        model([batch_Xₜ[1]])
        for (x, y) in zip(batch_Xₜ[2:end], batch_Xₜ₊₁[2:end])
            μ, logσ = model([x])
            σ = softplus(logσ)
            ŷ = mean(Float32.(rand(Normal(μ, softplus(logσ)), n_samples)))
            likelihood =
                -(log(sqrt(2 * π)) + log(σ) + ((y - ŷ)^2 / (2 * σ^2))) + likelihood
        end
        append!(likelihoods, likelihood / length(batch_Xₜ))
    end
    return likelihoods
end

# Load your data
col = :DE_KN_residential3_circulation_pump
data = DataFrame(CSV.read("./src/household_data_60min_singleindex.csv", DataFrame, types=Dict(:col => Float32), select=[col]))

#impute_mean(data, :DE_KN_residential3_circulation_pump)
data_clean = impute_with_surrounding_median(data, :DE_KN_residential3_circulation_pump, 10)

data_normalized, μ, cov = normalize_df!(data_clean)

@assert !any(ismissing, data_normalized[:, 1])

# Split the data into training and validation sets
train_data, val_data = split_data(data_normalized, 0.8)  # 80% for training, 20% for validation

# Define the model architecture
model = Chain(
    RNN(1 => 10, relu), RNN(10 => 10, relu), Dense(10 => 16, relu), Dense(16 => 2, identity)
)

# Set the hyperparameters
learning_rate = 1
batch_size = 40000
epochs = 10
n_mean = 10

deepAR_params = DeepARParams(; η=learning_rate, epochs=epochs, n_mean=n_mean)

loaderXtrain = DataLoader(train_data.DE_KN_residential3_circulation_pump[1:end-1], batchsize=batch_size)
loaderYtrain = DataLoader(train_data.DE_KN_residential3_circulation_pump[2:end], batchsize=batch_size)

# Train the model
loss = train_DeepAR(model, loaderXtrain, loaderYtrain, deepAR_params)

val_data_array = normalize_column!(val_data, col, μ, cov)
loaderXtrain = DataLoader(val_data_array[1:end-1], batchsize=batch_size)
loaderYtrain = DataLoader(val_data_array[2:end], batchsize=batch_size)
loss_eval = evaluate_model(model, loaderXtrain, loaderYtrain, 100)

ŷ = forescasting_DeepAR(model, train_data.DE_KN_residential3_circulation_pump, 100, 120; n_samples=100)
ŷ = moving_average(ŷ, 24)
