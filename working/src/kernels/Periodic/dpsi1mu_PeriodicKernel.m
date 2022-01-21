function dPsi1mu = dpsi1mu_PeriodicKernel(prs,X1,mu,sigma,xxQuad,wwQuad)
% function to compute Ey[k(y,x1)] where y ~ N(mu,sigma)
% X1 is N x 1 x M
% mu and sigma are T x 1 x M

[T,~,R] = size(mu);
M = size(X1,1); % column vector
Q = size(xxQuad,1); % quadrature nodes, column vector
assert(R == size(X1,3));

QuadEval = permute(sqrt(2*sigma).*xxQuad' + mu,[2 1 3]); % Q x T x R

QuadEval = reshape(QuadEval,[],1,size(mu,3)); % TQ x R

% gradient wrt first input argument
[~,GQuad] = dKin_PeriodicKernel(prs,QuadEval,X1); % TQ x M x TQ x R matrix 
% GQuad -- each slice contains grad wrt one input arg. using the chain

ddinmu = kron(speye(T),ones(Q,1)); % sparse rectangular matrix which picks out correct elements in GQuad for derivative wrt each mu, for each trial
% each column has Q ones at position corresponding to location of mu(t)


%%% old code 
GQuad = permute(reshape(permute(GQuad,[2 1 3]),M,Q,T,R),[3 1 4 2]);

dPsi1mu = sum(bsxfun(@times, GQuad, permute(wwQuad,[2 3 4 1])),4);

QuadEval = sqrt(2*sigma).*xxQuad' + mu; % T x Q x R

QuadEval = reshape(QuadEval,[],1,size(mu,3)); % TQ x R

GQuad = permute(reshape(GQuad,T,Q,M,[],R),[1 2 4 5 3]);

dPsi1mu = sum(bsxfun(@times, GQuad, permute(wwQuad,[2 3 4 5 1])),5);