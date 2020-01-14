function G = psi1_rbfKernel(prs,X1,mu,sigma,varargin)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

% hyperparameters
variance = 1;
lengthscale = prs(1);

if isempty(mu) && isempty(sigma)
    mu = zeros(0,1);
    sigma = zeros(0,1);
end
% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
    G = zeros(0,1);
else 
    % returns kernel Gram matrix
    ddist  = (mu - permute(X1,[2 1 3])).^2; % T x N x M
    G = variance^2*bsxfun(@times,exp(-0.5* (ddist./(sigma + lengthscale^2))),1./sqrt(sigma./lengthscale^2 + 1));

end

