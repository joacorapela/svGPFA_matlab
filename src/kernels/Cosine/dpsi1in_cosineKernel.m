function dGin2 = dpsi1in_cosineKernel(prs,X1,mu,sigma,varargin)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M
error('gradient of cosine kernel for inducing point locations not implemented')
% hyperparameters
variance = 1;
lengthscale = prs(1);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% returns kernel Gram matrix
ddist  = bsxfun(@minus,mu,permute(X1,[2 1 3])); % T x N x M 
ss = sigma./lengthscale^2 + 1; % T x 1 x M
G = variance^2*exp(-0.5*bsxfun(@times, ddist.^2,1./(lengthscale^2*ss)))./sqrt(ss); 

[N1,N2,ntr] = size(G);
dGin2 = zeros(N1,N2,N2,ntr);

ColMask = reshape(full(logical(kron(speye(N2),ones(N1,1)))),[N1 N2 N2]);
ColMask = repmat(ColMask,[1 1 1 ntr]);
dGin2(ColMask) = G.* ddist./(lengthscale^2*ss);