# spikeGPFA: sparse variational GPFA for spike train data

This repository contains a toolbox for fitting Gaussian Process Factor Analysis models to spike train data

## Supported Likelihoods
* Poisson likelihood
* Point Process likelihood

## Supported Models

### Sparse variational GPFA (svGPFA)
A classic GPFA model for spike train data.

## Supported Covariance Functions:
* Exponentiated Quadratic Kernel (RBF)
* Periodic Kernel
* Cosine Kernel
* Locally Periodic Kernel
* Rational Quadratic Kernel
* Matern 3/2 Kernel
* Matern 5/2 Kernel
* Linear Kernel
* RBF + Linear Composition Kernel
* Periodic + Linear Composition Kernel
* Locally Periodic + Linear Composition Kernel

## Supported Output Non-Linearities:
* exponential
* linear rectified
* soft recitified
* sigmoid
* custom (just implement the non-linearity and gradient wrt to the input argument)

## Basic Usage:
You need to pick the number of inducing points, and kernel functions and then initialise a model structure using the InitialiseModel_svGPFA.m function. You can then pass this structure into the variationalEM.m function which will optimise the model. See ./demos for some examples.

## Dependencies:
* Matlab 2016b or newer
* mtimesx (in util)
* minFunc (in util)

