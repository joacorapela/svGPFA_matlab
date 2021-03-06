function [dGin2, dGin1] = dKin_PeriodicKernel(prs,X1,varargin)

variance = 1;
lengthscale = prs(1);
period = prs(2);

assert(length(prs) == 2)

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% inputs
if nargin == 2
    X2 = X1;
else
    X2 = varargin{1};
    if isempty(X2)
        X2 = zeros(0,1);
    end
end

G = PeriodicKernel(prs,X1,X2);
dd = bsxfun(@minus,X1,permute(X2,[2 1 3]));
rr = (pi.*dd./period);

dGin2 = zeros([size(G,1),size(G,2),size(X2,1), size(G,3)]);

if nargout > 1
    dGin1 = zeros([size(G,1),size(G,2),size(X1,1), size(G,3)]);
end

for ii = 1:size(X2,1)
    dGin2(:,ii,ii,:) = permute(4*pi/period*G(:,ii,:)/lengthscale^2 .* ...
        sin(rr(:,ii,:)).*cos(rr(:,ii,:)),[1 2 4 3]);
end


% gradient with respect to input points X1
if nargout > 1
    for ii = 1:size(X1,1)
        if nargin == 2 % grad wrt to first input if same
            dGin1(ii,:,ii,:) = permute(-4*pi/period*G(ii,:,:)/lengthscale^2 .* ...
                sin(rr(ii,:,:)).*cos(rr(ii,:,:)),[1 2 4 3]);
            dGin1(:,ii,ii,:) = permute(4*pi/period*G(:,ii,:)/lengthscale^2 .* ...
                sin(rr(:,ii,:)).*cos(rr(:,ii,:)),[1 2 4 3]);
        else % grad wrt first input if different
            dGin1(ii,:,ii,:) = permute(-4*pi/period*G(ii,:,:)/lengthscale^2 .* ...
                sin(rr(ii,:,:)).*cos(rr(ii,:,:)),[1 2 4 3]);
        end
    end
end



