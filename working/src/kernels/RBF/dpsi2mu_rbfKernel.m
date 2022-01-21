function dGmu = dpsi2mu_rbfKernel(prs,X1,mu,sigma,varargin)
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
ss = permute((2 * sigma./lengthscale^2 + 1),[2 4 1 3]); % 1 x 1 x T x M
ddist  = (X1 - permute(X1,[2 1 3])).^2; % N x N x M
xplus = (X1 + permute(X1,[2 1 3]))/2; % N x N x M
mudist = (permute(mu,[4 2 1 3]) - permute(xplus,[1 2 4 3])); % N x N x T x M

if nargin < 5
    G = variance^4./sqrt(ss)...
        .* exp(- (permute(ddist,[1 2 4 3])/(4*lengthscale^2) +  mudist.^2 ./(lengthscale^2*ss))); % N x N x T x M
else
    G = varargin{1};
end
dGmu = G .*(-2 * mudist ./(lengthscale^2*ss)); % N x N x T x M