function [Gdiag,dGdiag] = Kdiag_rbf_and_linearKernel(prs,X1)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential

variance_rbf = prs(1);
variance_lin = prs(2);
rbfprs = prs(3);
linearprs = prs(4:end);

[Gdiag1,dGdiag1] = Kdiag_rbfKernel(rbfprs,X1);
[Gdiag2,dGdiag2] = Kdiag_LinearKernel(linearprs,X1);

Gdiag = variance_rbf^2 * Gdiag1 + variance_lin^2 * Gdiag2;


dGdiag0(:,:,1,:) = permute(2 * variance_rbf * Gdiag1,[1 2 4 3]);
dGdiag0(:,:,2,:) = permute(2 * variance_lin * Gdiag2,[1 2 4 3]); 

dGdiag = cat(3,dGdiag0,variance_rbf^2 * dGdiag1,variance_lin^2 * dGdiag2);

end