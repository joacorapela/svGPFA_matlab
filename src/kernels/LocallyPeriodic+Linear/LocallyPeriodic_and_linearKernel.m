function G = LocallyPeriodic_and_linearKernel(prs,X1,varargin)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential
%
% hyperparameters
variance_loc = prs(1);
variance_lin = prs(2);
locperprs = prs(3:5);
linearprs = prs(6:end);

if nargin > 2
    G = variance_loc^2 * LocallyPeriodicKernel(locperprs,X1,varargin{1}) + variance_lin^2 * LinearKernel(linearprs,X1,varargin{1});
else
    G = variance_loc^2 * LocallyPeriodicKernel(locperprs,X1) + variance_lin^2 * LinearKernel(linearprs,X1);
end

end

