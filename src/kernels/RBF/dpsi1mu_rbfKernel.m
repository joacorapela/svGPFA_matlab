function dGin1 = dpsi1mu_rbfKernel(prs,X1,mu,sigma,varargin)
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
ddist  = mu - permute(X1,[2 1 3]); % T x N x M 
ss = sigma./lengthscale^2 + 1; %  T x 1c x M
G = variance^2*exp(-0.5*(ddist.^2./(lengthscale^2*ss)))./sqrt(ss); 

dGin1 = - G.* ddist./(lengthscale^2*ss);