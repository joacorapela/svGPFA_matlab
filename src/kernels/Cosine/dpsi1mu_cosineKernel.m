function dGin1 = dpsi1mu_cosineKernel(prs,X1,mu,sigma,varargin)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

% hyperparameters
variance = 1;
lengthscale = prs(1);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% returns kernel Gram matrix
ddist  = bsxfun(@minus,mu,permute(X1,[2 1 3])); % T x N x M 
dGin1 = - variance^2./lengthscale^2 .* exp(-sigma./(2*lengthscale^4)) .* sin(ddist./lengthscale^2);