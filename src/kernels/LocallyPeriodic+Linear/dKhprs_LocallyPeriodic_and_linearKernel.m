function [dKhprs] = dKhprs_LocallyPeriodic_and_linearKernel(prs,X1,varargin)
% get hprs for each kernel
variance_loc = prs(1);
variance_lin = prs(2);

locperprs = prs(3:5);
linearprs = prs(6:end);

% get gradients
if nargin > 2
    dKhprs1 = dKhprs_LocallyPeriodicKernel(locperprs,X1,varargin{1});
    dKhprs2 = dKhprs_LinearKernel(linearprs,X1,varargin{1});
    dKhprs0(:,:,1,:) = 2 * variance_loc * LocallyPeriodicKernel(locperprs,X1,varargin{1});
    dKhprs0(:,:,2,:) = 2 * variance_lin * LinearKernel(linearprs,X1,varargin{1});
else
    dKhprs1 = dKhprs_LocallyPeriodicKernel(locperprs,X1);
    dKhprs2 = dKhprs_LinearKernel(linearprs,X1);
    dKhprs0(:,:,1,:) = 2 * variance_loc * LocallyPeriodicKernel(locperprs,X1);
    dKhprs0(:,:,2,:) = 2 * variance_lin * LinearKernel(linearprs,X1);
end
% concatenate grads
dKhprs = cat(3,dKhprs0,variance_loc.^2*dKhprs1,variance_lin.^2*dKhprs2);
