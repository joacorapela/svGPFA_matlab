function G = psi1_cosineKernel(prs,X1,mu,sigma,varargin)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

% hyperparameters
variance = 1;
lengthscale = prs(1);

% take care of empty input
if isempty(X1)
    G = zeros(0,1);
else 
    % returns kernel Gram matrix
    ddist  = bsxfun(@minus,mu,permute(X1,[2 1 3])); % T x N x M
    G = variance^2*cos(ddist./lengthscale^2).*exp(-sigma./(2*lengthscale^4));
end