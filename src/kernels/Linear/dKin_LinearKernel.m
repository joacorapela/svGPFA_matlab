function [dGin2, dGin1] = dKin_LinearKernel(prs,X1,varargin)

% gradient with respect to inputs

% hyperparameters
variance = 1;
slope = prs(1);
centre = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% inputs
if nargin == 2
    X2 = X1;
else
    X2 = varargin{1};
    if isempty(X2)
        X2 = zeros(0,1);
    end
end

G = variance^2 + slope^2 * bsxfun(@times,X1 - centre,permute(X2 - centre,[2 1 3]));

[N1,N2,ntr] = size(G);

dGin2 = zeros(N1,N2,N2,ntr);

if nargout > 1
    dGin1 = zeros(N1,N2,N1,ntr);
end

ColMask = reshape(full(logical(kron(speye(N2),ones(N1,1)))),[N1 N2 N2]);
ColMask = repmat(ColMask,[1 1 1 ntr]);
dGin2(ColMask) = slope^2 * (X1 - centre);

% gradient with respect to input points X1
if nargout > 1
    RowMask = reshape(full(logical(kron(speye(N1),ones(1,N2)))),[N1 N2 N1]);
    RowMask = repmat(RowMask,1,1,1,ntr);
    if nargin == 2 % grad wrt to first input if same
        dGin1(RowMask) = permute(slope^2 * (X2 - centre),[2 1 3 4]);
        dGin1(ColMask) = slope^2 * (X2 - centre);
    else % grad wrt first input if different
        dGin1(RowMask) = slope^2 * (X2 - centre);
    end
end



