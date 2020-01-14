function G = PeriodicKernel(prs,X1,varargin)
% periodic kernel covariance function
variance = 1;
lengthscale = prs(1);
period = prs(2);

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

% returns kernel Gram matrix and derivative wrt hyperparameters
dd = bsxfun(@minus,X1,permute(X2,[2 1 3]));

rr = (pi.*dd./period);
G = variance^2*exp(-2*sin(rr).^2/lengthscale^2);

% filename = '~/dev/research/gatsby/svGPFA/code/test/data/PeriodicKernel.mat';
% save(filename, 'X1', 'lengthscale', 'period', 'G');
% 
% keyboard
