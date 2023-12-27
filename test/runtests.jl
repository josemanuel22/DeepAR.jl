using DeepAR
using Test
using Flux
using Distributions
using Flux.Data: DataLoader

# Test setup
#Base.@kwdef mutable struct DeepARParams
#    η = 1e-2
#    epochs::Int = 10
#    n_mean::Int = 100
#end

#Flux.setup(::Flux.Optimiser, m::MockDeepARModel) = m
#Flux.reset!(m::MockDeepARModel) = nothing
#(m::MockDeepARModel)(x) = (m.params[1], log(abs(m.params[1])))

# ... (train_DeepAR function definition)

# Test cases
@testset "DeepAR Training Tests" begin
    # Mock a simple model for testing
    model = Chain(
        RNN(1 => 10, relu), RNN(10 => 10, relu), Dense(10 => 16, relu), Dense(16 => 2, identity)
    )
    hparams = DeepARParams()

    # Test 1: Functionality with non-empty loaders
    Xtrain = Float32.(randn(1, 1001))
    Ytrain = Xtrain[2:1001]
    Xtrain = Xtrain[1:1000]
    loaderXtrain = DataLoader(Xtrain, batchsize=100, shuffle=false)
    loaderYtrain = DataLoader(Ytrain, batchsize=100, shuffle=false)

    losses = train_DeepAR(model, loaderXtrain, loaderYtrain, hparams)
    @test length(losses) == hparams.epochs
    @test all(x -> x <= 0, losses)  # Assuming loss should be negative or zero

    # Test 2: Handling of empty loaders
    empty_loader = DataLoader(Float32.(randn(1, 1000)), batchsize=100, shuffle=false)
    losses_empty = train_DeepAR(model, empty_loader, empty_loader, hparams)
    @test !isempty(losses_empty)

    empty_loader = DataLoader(Float32.(randn(0, 1000)), batchsize=100, shuffle=false)
    @test_throws Exception train_DeepAR(model, empty_loader, empty_loader, hparams)

    # Test 3: Error handling for invalid inputs
    @test_throws Exception train_DeepAR(model, "invalid", loaderYtrain, hparams)
    @test_throws Exception train_DeepAR(model, loaderXtrain, "invalid", hparams)

    # Test 4: Non-default hyperparameters
    Xtrain = Float32.(randn(1, 1001))
    Ytrain = Xtrain[2:1001]
    Xtrain = Xtrain[1:1000]
    loaderXtrain = DataLoader(Xtrain, batchsize=200, shuffle=false)
    loaderYtrain = DataLoader(Ytrain, batchsize=200, shuffle=false)
    custom_hparams = DeepARParams(η=1e-3, epochs=5, n_mean=50)
    losses_custom = train_DeepAR(model, loaderXtrain, loaderYtrain, custom_hparams)
    @test length(losses_custom) == custom_hparams.epochs
end
