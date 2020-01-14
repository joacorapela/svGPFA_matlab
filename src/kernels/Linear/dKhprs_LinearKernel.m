function [dKhprs] = dKhprs_LinearKernel(prs,X1,varargin)
%
% gradient with respect to hyperparameters
%
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

% returns kernel Gram matrix
G = variance^2 + slope^2 * bsxfun(@times,X1 - centre,permute(X2 - centre,[2 1 3]));

% dKhprs(:,:,1,:) = 2*variance*ones(size(G));
dKhprs(:,:,1,:) = 2 *(G - variance^2)/slope;
dKhprs(:,:,2,:) = 2*slope^2 * centre - slope^2 * bsxfun(@plus,X1,permute(X2,[2 1 3]));

