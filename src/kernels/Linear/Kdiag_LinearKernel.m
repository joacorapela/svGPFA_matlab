function [Gdiag,dGdiag] = Kdiag_LinearKernel(prs,X1)
% linear kernel
%
% hyperparameters
variance = 1;
slope = prs(1);
centre = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end
% computes only diagonal variance elements of kernel with equal
% inputs
Gdiag = variance^2 + slope^2 * (X1 - centre).^2;

% dGdiag(:,:,1,:) = permute(2*variance*ones(size(Gdiag)),[1 2 4 3]);
dGdiag(:,:,1,:) = permute(2 * slope * (X1 - centre).^2,[1 2 4 3]);
dGdiag(:,:,2,:) = permute(-2 * slope^2 * (X1 - centre),[1 2 4 3]);
end