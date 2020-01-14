function [dKhprs] = dKhprs_Periodic_and_linearKernel(prs,X1,varargin)
% get hprs for each kernel
variance_per = prs(1);
variance_lin = prs(2);
periodprs = prs(3:4);
linearprs = prs(5:end);
% get gradients
if nargin > 2
    dKhprs1 = dKhprs_PeriodicKernel(periodprs,X1,varargin{1});
    dKhprs2 = dKhprs_LinearKernel(linearprs,X1,varargin{1});
    
    dKhprs0(:,:,1,:) = 2 * variance_per * PeriodicKernel(periodprs,X1,varargin{1});
    dKhprs0(:,:,2,:) = 2 * variance_lin * LinearKernel(linearprs,X1,varargin{1});
    
else
    dKhprs1 = dKhprs_PeriodicKernel(periodprs,X1);
    dKhprs2 = dKhprs_LinearKernel(linearprs,X1);
    
    dKhprs0(:,:,1,:) = 2 * variance_per * PeriodicKernel(periodprs,X1);
    dKhprs0(:,:,2,:) = 2 * variance_lin * LinearKernel(linearprs,X1);
end
% concatenate grads
dKhprs = cat(3,dKhprs0,variance_per^2*dKhprs1,variance_lin^2*dKhprs2);
