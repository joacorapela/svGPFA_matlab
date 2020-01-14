function dGhprs = dpsi1hprs_cosineKernel(prs,X1,mu,sigma,varargin)
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

% dGhprs(:,:,1,:) = 2*G/variance; % N x T x M
dGhprs(:,:,1,:) = variance^2 * exp(-sigma./(2*lengthscale^4)).*(2*sigma./lengthscale^5 .* cos(ddist./lengthscale^2) ...
    + 2*ddist./lengthscale^3 .* sin(ddist./lengthscale^2));

