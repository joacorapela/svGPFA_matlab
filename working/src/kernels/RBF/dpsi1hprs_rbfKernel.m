function dGhprs = dpsi1hprs_rbfKernel(prs,X1,mu,sigma,varargin)
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
ddist  = bsxfun(@minus,mu,permute(X1,[2 1 3])).^2; % T x N x M 
ss = sigma./lengthscale^2 + 1; % 1 x T x M
G = variance^2*exp(-0.5*bsxfun(@times, ddist,1./(lengthscale^2*ss)))./sqrt(ss); 

% dGhprs(:,:,1,:) = 2*G/variance; % N x T x M
dGhprs(:,:,1,:) = G .* (sigma/lengthscale^3 * 1./ ss + lengthscale * bsxfun(@times, ddist,1./(lengthscale^2 + sigma).^2));