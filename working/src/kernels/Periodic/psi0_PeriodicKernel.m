function [Psi0,dPsi0] = psi0_PeriodicKernel(prs,X1,mu,sigma,varargin)

variance = 1;
lengthscale = prs(1);
period = prs(2);

Psi0 = variance^2*ones(size(X1));

% dPsi0(:,:,1,:) = permute(2*Psi0/variance,[1 2 4 3]);
dPsi0(:,:,1,:) = permute(zeros(size(Psi0)),[1 2 4 3]);
dPsi0(:,:,2,:) = permute(zeros(size(Psi0)),[1 2 4 3]);
