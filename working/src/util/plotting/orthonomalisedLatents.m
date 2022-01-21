function [mu_fs_orth,Corth] = orthonomalisedLatents(C,mu_fs)
% function to predict latent function given model fit and orthonormalize
% basis for consistency. 
% input:
% m  -  is a structure contianing the model
% t  -  is a row vector of time points where GP should be evaluated
% output:
% mu_fs_orth
%
%
[U,S,V] = svd(C);
dx = size(C,2);

mu_fs_orth = mtimesx(mu_fs,V*(S(1:dx,:)'));

Corth = U(:,1:dx);
