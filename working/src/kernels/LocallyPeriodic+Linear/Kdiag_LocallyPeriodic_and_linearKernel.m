function [Gdiag,dGdiag] = Kdiag_LocallyPeriodic_and_linearKernel(prs,X1)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential

variance_loc = prs(1);
variance_lin = prs(2);
locperprs = prs(3:5);
linearprs = prs(6:end);

[Gdiag1,dGdiag1] = Kdiag_LocallyPeriodicKernel(locperprs,X1);
[Gdiag2,dGdiag2] = Kdiag_LinearKernel(linearprs,X1);

Gdiag = Gdiag1 + Gdiag2;

dGdiag0(:,:,1,:) = 2 * variance_loc * Gdiag1;
dGdiag0(:,:,2,:) = 2 * variance_lin * Gdiag2;

dGdiag = cat(3,dGdiag0,variance_loc^2 * dGdiag1,variance_lin^ 2 * dGdiag2);

end