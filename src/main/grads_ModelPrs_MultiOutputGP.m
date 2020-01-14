function [ddmu_h, ddvar_h] = grads_ModelPrs_MultiOutputGP(m,C,b,mu_k,var_k)
% function to compute gradients of 
% mu_h = C* mu_k + b
% var_h = C^2 var_k

% gradients with respect to C
ddmu_h{1} = repmat(permute(mu_k,[4 2 1 3]),size(C,1),1,1,1);
ddvar_h{1} = 2*bsxfun(@times,C,permute(var_k,[4 2 1 3]));

% gradients with respect to b
ddmu_h{2} = ones(length(b),size(mu_k,2),size(mu_k,1),size(mu_k,3)); 
