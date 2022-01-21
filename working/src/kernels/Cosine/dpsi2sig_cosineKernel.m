function dGsig = dpsi2sig_cosineKernel(prs,X1,mu,sigma,varargin)
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
xplus = bsxfun(@plus,X1,permute(X1,[2 1 3])); % N x N x M
mudist = bsxfun(@minus,permute(2*mu,[4 2 1 3]),permute(xplus,[1 2 4 3])); % N x N x T x M
dGsig = - variance^2/lengthscale^4 * exp(-2*permute(sigma,[2 4 1 3])./lengthscale^4) .* cos(mudist./lengthscale^2);
