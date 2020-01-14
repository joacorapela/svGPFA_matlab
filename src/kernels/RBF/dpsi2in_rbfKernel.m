function dGin = dpsi2in_rbfKernel(prs,X1,mu,sigma,varargin)
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
ddist  = permute(bsxfun(@minus,X1,permute(X1,[2 1 3])),[1 2 4 3]); % N x N x 1 x M
xplus = bsxfun(@plus,X1,permute(X1,[2 1 3]))/2; % N x N x M
mudist = bsxfun(@minus,permute(mu,[4 2 1 3]),permute(xplus,[1 2 4 3])); % N x N x T x M

G = variance^4./sqrt(ss)...
    .* exp(- ddist.^2/(4*lengthscale^2) - mudist.^2./(lengthscale^2*ss)); % N x N x T x M

[N1,N2,T,ntr] = size(G);

G = permute(G,[1 2 3 5 4]);
ddist = permute(ddist,[1 2 3 5 4]);
mudist = permute(mudist,[1 2 3 5 4])./(lengthscale^2*permute(ss,[1 2 3 5 4]));

ColMask = reshape(full((kron(speye(N2),ones(N1,1)))),[N1 N2 1 N2]);
ColMask = repmat(ColMask,[1 1 T 1 ntr]);
RowMask = reshape(full((kron(speye(N1),ones(1,N2)))),[N1 N2 1 N1]);
RowMask = repmat(RowMask,[1 1 T 1 ntr]);

dGin = G .* (-0.5 * ddist/(lengthscale^2) .* RowMask ...
    + 0.5 * ddist/(lengthscale^2) .* ColMask ...
    + mudist.* RowMask + mudist .* ColMask);


% TO DO: DO this with logical indexing instead... need to speed this up
% significantly!!

