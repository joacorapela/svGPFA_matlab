function [Gdiag,dGdiag] = psi0_cosineKernel(prs,X1,varargin)

% hyperparameters
variance = 1;
lengthscale = prs(1);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end
% computes only diagonal variance elements of kernel with equal
% inputs
Gdiag = variance^2*ones(size(X1));
% dGdiag(:,:,1,:) = permute(2*Gdiag/variance,[1 2 4 3]);
dGdiag(:,:,1,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
end