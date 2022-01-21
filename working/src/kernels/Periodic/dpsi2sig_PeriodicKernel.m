function dGsig = dpsi2sig_PeriodicKernel(prs,X1,mu,sigma,xxQuad,wwQuad)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

% hyperparameters
variance = 1;
lengthscale = prs(1);
period = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% returns kernel Gram matrix

ss = permute((2 * sigma./lengthscale^2 + 1),[2 4 1 3]); % 1 x 1 x T x M
ddist  = bsxfun(@minus,X1,permute(X1,[2 1 3])).^2; % N x N x M
xplus = bsxfun(@plus,X1,permute(X1,[2 1 3]))/2; % N x N x M
mudist = bsxfun(@minus,permute(mu,[4 2 1 3]),permute(xplus,[1 2 4 3])); % N x N x T x M

G = variance^4./sqrt(ss)...
    .* exp(- bsxfun(@plus,permute(ddist,[1 2 4 3])/(4*lengthscale^2), mudist.^2 ./(lengthscale^2*ss))); % N x N x T x M

dGsig = G .* (2 * mudist.^2./(2 *permute(sigma,[2 4 1 3]) + lengthscale^2).^2 ...
    - 1 ./ (permute(2 * sigma,[2 4 1 3]) + lengthscale^2));