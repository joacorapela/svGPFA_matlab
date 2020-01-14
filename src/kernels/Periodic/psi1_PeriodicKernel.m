function Psi1 = psi1_PeriodicKernel(prs,X1,mu,sigma,xxQuad,wwQuad)
%
% compute Psi1 statistic using Gauss Hermite quadrature weights
%
% X1 --- M x 1 x R 
% mu,sigma -- T x 1 x R
% xxQuad -- Q x 1
% wwQuad -- Q x 1

[T,~,R] = size(mu);
M = size(X1,1); % column vector
Q = size(xxQuad,1); % quadrature nodes, column vector
assert(R == size(X1,3));

QuadEval = permute(sqrt(2*sigma).*xxQuad' + mu,[2 1 3]); % Q x T x R

QuadEval = reshape(QuadEval,[],1,size(mu,3)); % TQ x R

GQuad = PeriodicKernel(prs,QuadEval,X1); % TQ x M x R matrix

GQuad = permute(reshape(permute(GQuad,[2 1 3]),M,Q,T,R),[3 1 4 2]);

Psi1 = sum(bsxfun(@times, GQuad, permute(wwQuad,[2 3 4 1])),4);

