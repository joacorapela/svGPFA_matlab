function [Gdiag,dGdiag] = Kdiag_Periodic_and_linearKernel(prs,X1)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential

variance_per = prs(1);
variance_lin = prs(2);
periodprs = prs(3:4);
linearprs = prs(5:end);

[Gdiag1,dGdiag1] = Kdiag_PeriodicKernel(periodprs,X1);
[Gdiag2,dGdiag2] = Kdiag_LinearKernel(linearprs,X1);

Gdiag = variance_per^2*Gdiag1 + variance_lin^2*Gdiag2;

dGdiag0(:,:,1,:) = 2*variance_per*Gdiag1;
dGdiag0(:,:,2,:) = 2*variance_lin*Gdiag2;

dGdiag = cat(3,dGdiag0,variance_per^2*dGdiag1,variance_lin^2*dGdiag2);

end