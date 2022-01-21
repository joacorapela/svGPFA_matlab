function lik_poi = expectedLogLik_Poisson(m,mu_h,var_h,ntr);

if nargin < 4
    trEval = 1:m.ntr;
else
    trEval = ntr;
end

% account for zero padding
mask = permute(repmat(m.mask(:,trEval),1,1,m.dy),[1 3 2]);
mu_h(mask) = 0; % account for zero padding
var_h(mask) = 0;

if strcmp(func2str(m.link),'exponential') % use closed form expectations
    intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
    log_link = mu_h;
else
    intval = permute(mtimesx(m.wwHerm',permute(m.link(mu_h + sqrt(2*var_h).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);
    log_link =permute(mtimesx(m.wwHerm',permute(log(m.link(mu_h + sqrt(2*var_h).* permute(m.xxHerm,[2 3 4 1]))),[4 1 2 3])),[2 3 4 1]); 
end
lik_poi1 = m.BinWidth*sum(intval(:));
lik_poi2 = sum(vec(m.Y(:,:,trEval).*permute(log_link,[2 1 3]))); 

lik_poi = -lik_poi1 + lik_poi2; 

% lik_poi = lik_poi2; 


% - sum(log(factorial(vec(m.Y(:,:,trEval))))) +
% log(m.BinWidth)*sum(vec(m.Y(:,:,trEval))); ignore constants -- too
% expensive to compute factorials 

