# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

[![Build Status](https://github.com/josemanuel22/DeepAR.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/josemanuel22/DeepAR.jl/actions/workflows/CI.yml?query=branch%3Amain)


DeepAR is a Julia module designed for probabilistic forecasting using autoregressive recurrent networks, based on the concept introduced in the paper [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/pdf/1704.04110.pdf).

## Installation
Currently, DeepAR is not registered in Julia's General registry. To install, clone the repository directly.

## Features
DeepAR allows the training and use of deep learning models for time series forecasting. Key features include:

- Time series forecasting using autoregressive models.
- Training and prediction with custom hyperparameters.

## Usage

### Importing the Module
To use DeepAR, first import the module along with required dependencies:
```julia
using DeepAR
```

### Hyperparameters
Define the hyperparameters for the DeepAR model using `DeepARParams`:
- `η::Float64`: Learning rate for model training. Default is `1e-2`.
- `epochs::Int`: Number of training epochs. Default is `10`.
- `n_mean::Int`: Number of samples used for computing the predictive mean. Default is `100`.

### Training
Train a DeepAR model with the `train_DeepAR` function:
- Parameters:
  - `model`: The DeepAR model.
  - `loaderXtrain`: DataLoader with input sequences.
  - `loaderYtrain`: DataLoader with target sequences.
  - `hparams`: `DeepARParams` instance.
- Returns a vector of loss values.

### Forecasting
Generate forecasts using the `forecasting_DeepAR` function:
- Parameters:
  - `model`: Trained DeepAR model.
  - `ts`: Time series data.
  - `t₀`: Starting time step for forecasting.
  - `τ`: Number of time steps to forecast.
  - `n_samples`: Number of samples per forecast (default: 100).
- Returns a vector of forecasted values.

## Contributing
Contributions to the DeepAR module are welcome. Please submit issues and pull requests on the repository.

## License
DeepAR is licensed under MIT. Please check the repository for more details.

---

For more detailed information and updates, refer to the DeepAR GitHub repository.
