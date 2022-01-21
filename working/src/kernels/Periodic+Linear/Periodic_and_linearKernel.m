function G = Periodic_and_linearKernel(prs,X1,varargin)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential
%
% hyperparameters

variance_per = prs(1);
variance_lin = prs(2);
periodprs = prs(3:4);
linearprs = prs(5:end);

if nargin >2
    G = variance_per^2 * PeriodicKernel(periodprs,X1,varargin{1}) + variance_lin^2 * LinearKernel(linearprs,X1,varargin{1});
else
    G = variance_per^2 * PeriodicKernel(periodprs,X1) + variance_lin^2 * LinearKernel(linearprs,X1);
end

end

