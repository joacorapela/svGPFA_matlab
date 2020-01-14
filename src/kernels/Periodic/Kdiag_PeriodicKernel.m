function [Gdiag,dGdiag] = Kdiag_PeriodicKernel(prs,X1)
% computes only diagonal variance elements of kernel with equal
% inputs
variance = 1;
lengthscale = prs(1);
period = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

Gdiag = variance^2*ones(size(X1));
% dGdiag(:,:,1,:) = permute(2*Gdiag/variance,[1 2 4 3]);
dGdiag(:,:,1,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
dGdiag(:,:,2,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);

% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/master/ci/data/Kdiag_PeriodicKernel.mat';
% save(filename, 'X1', 'variance', 'lengthscale', 'period', 'Gdiag');

% keyboard
