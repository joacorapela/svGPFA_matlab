function G = rbf_and_linearKernel(prs,X1,varargin)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential
%
% hyperparameters
variance_rbf = prs(1);
variance_lin = prs(2);
rbfprs = prs(3);
linearprs = prs(4:end);

if nargin > 2
    G = variance_rbf^2 * rbfKernel(rbfprs,X1,varargin{1}) + variance_lin^2 * LinearKernel(linearprs,X1,varargin{1});
else
    G = variance_rbf^2 * rbfKernel(rbfprs,X1) + variance_lin^2 * LinearKernel(linearprs,X1);
end

end

