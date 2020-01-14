function [dKhprs] = dKhprs_cosineKernel(prs,X1,varargin)
%
% gradient with respect to hyperparameters
%
% hyperparameters
variance = 1;
period = prs(1);

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
ddist  = bsxfun(@minus,X1,permute(X2,[2 1 3]));

dKhprs(:,:,1,:) = 2*variance^2*sin(ddist/period^2)/period^3 .* ddist;

