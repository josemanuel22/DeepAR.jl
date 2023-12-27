module DeepAR

using Flux
using StatsBase
using Random
using ProgressMeter
using Distributions

"""
DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
https://arxiv.org/pdf/1704.04110.pdf
"""

"""
DeepArParams

Hyperparameters for the DeepAR model.

- `η::Float64`: The learning rate for model training. Default is `1e-2`.
- `epochs::Int`: The number of training epochs. Default is `10`.
- `n_mean::Int`: The number of samples used for computing the predictive mean. Default is `100`.
"""
Base.@kwdef mutable struct DeepARParams
    η = 1e-2
    epochs::Int = 10
    n_mean::Int = 100
end

"""
train_DeepAR(model, loaderXtrain, loaderYtrain, hparams)

Train a DeepAR model using the given data loaders and hyperparameters.

- `model`: The DeepAR model to be trained.
- `loaderXtrain`: DataLoader containing input sequences for training.
- `loaderYtrain`: DataLoader containing target sequences for training.
- `hparams`: An instance of `DeepARParams` specifying hyperparameters for training.

Returns a vector of loss values during training.
"""
function train_DeepAR(model, loaderXtrain, loaderYtrain, hparams::DeepARParams)
    # Assert that DataLoaders are not empty
    # Check if the DataLoaders have data
    if size(loaderXtrain.data, 1) == 0
        throw(ArgumentError("loaderXtrain is empty and has no data to iterate over"))
    end
    if size(loaderYtrain.data, 1) == 0
        throw(ArgumentError("loaderYtrain is empty and has no data to iterate over"))
    end

    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), model)
    @showprogress for (batch_Xₜ, batch_Xₜ₊₁) in zip(loaderXtrain, loaderYtrain)
        loss, grads = Flux.withgradient(model) do m
            likelihood = 0
            Flux.reset!(m)
            model([batch_Xₜ[1]])
            for (x, y) in zip(batch_Xₜ[2:end], batch_Xₜ₊₁[2:end])
                μ, logσ = model([x])
                σ = softplus(logσ)
                ŷ = mean(μ .+ σ .* Float32.(rand(Normal(μ, σ), hparams.n_mean)))
                likelihood =
                    -(log(sqrt(2 * π)) + log(σ) + ((y - ŷ)^2 / (2 * σ^2))) + likelihood
            end
            likelihood / length(batch_Xₜ)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
    return losses
end

"""
forecasting_DeepAR(model, ts, t₀, τ; n_samples=100)

Generate forecasts using a trained DeepAR model.

- `model`: The trained DeepAR model.
- `ts`: Time series data used for forecasting.
- `t₀`: Time step at which forecasting starts.
- `τ`: Number of time steps to forecast.
- `n_samples`: Number of samples to draw for each forecast (default is 100).

Returns a vector containing the forecasted values.
"""
function forescasting_DeepAR(model, ts, t₀, τ; n_samples=100)
    prediction = []
    Flux.reset!(model)
    μ, logσ = 0.0f0, 0.0f0
    for x in ts[1:t₀]
        μ, logσ = model([x])
    end

    for x in ts[(t₀+1):(t₀+τ)]
        ŷ = mean(Float32.(rand(Normal(μ, softplus(logσ)), n_samples)))
        μ, logσ = model([ŷ])
        append!(prediction, ŷ)
    end
    return prediction
end

export forescasting_DeepAR, train_DeepAR, DeepARParams

end # module
