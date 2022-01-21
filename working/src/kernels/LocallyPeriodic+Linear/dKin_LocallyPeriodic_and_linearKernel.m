function [dGin2, dGin1] = dKin_LocallyPeriodic_and_linearKernel(prs,X1,varargin)

variance_loc = prs(1);
variance_lin = prs(2);

locperprs = prs(3:5);
linearprs = prs(6:end);

if nargin > 2
    if nargout > 1
        [dGin21,dGin11] = dKin_LocallyPeriodicKernel(locperprs,X1,varargin{1});
        [dGin22,dGin12] = dKin_LinearKernel(linearprs,X1,varargin{1});
        
        dGin2 = variance_loc.^2 * dGin21 + variance_lin.^2 * dGin22;
        dGin1 = variance_loc.^2 * dGin11 + variance_lin.^2 * dGin12;
        
    else
        dGin21 = dKin_LocallyPeriodicKernel(locperprs,X1,varargin{1});
        dGin22 = dKin_LinearKernel(linearprs,X1,varargin{1});
        
        dGin2 = variance_loc.^2 * dGin21 + variance_lin.^2 * dGin22;
        
    end
else
    if nargout > 1
        [dGin21,dGin11] = dKin_LocallyPeriodicKernel(locperprs,X1);
        [dGin22,dGin12] = dKin_LinearKernel(linearprs,X1);
        
        dGin2 = variance_loc.^2 * dGin21 + variance_lin.^2 * dGin22;
        dGin1 = variance_loc.^2 * dGin11 + variance_lin.^2 * dGin12;
        
    else
        dGin21 = dKin_LocallyPeriodicKernel(locperprs,X1);
        dGin22 = dKin_LinearKernel(linearprs,X1);
        
        dGin2 = variance_loc.^2 * dGin21 + variance_lin.^2 * dGin22;
        
    end
end