function dPsi1hprs = dpsi1hprs_PeriodicKernel(prs,X1,mu,sigma,xxQuad,wwQuad)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

[T,~,R] = size(mu);
M = size(X1,1); % column vector
Q = size(xxQuad,1); % quadrature nodes, column vector
assert(R == size(X1,3));

QuadEval = sqrt(2*sigma).*xxQuad' + mu; % T x Q x R

QuadEval = reshape(QuadEval,[],1,size(mu,3)); % TQ x R

GQuad = dKhprs_PeriodicKernel(prs,QuadEval,X1); % TQ x M x numhprs x R matrix

GQuad = permute(reshape(GQuad,T,Q,M,[],R),[1 2 4 5 3]);

dPsi1hprs = sum(bsxfun(@times, GQuad, permute(wwQuad,[2 3 4 5 1])),5);
