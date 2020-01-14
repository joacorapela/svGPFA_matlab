function G = LinearKernel(prs,X1,varargin)
% linear kernel covariance function
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

end

