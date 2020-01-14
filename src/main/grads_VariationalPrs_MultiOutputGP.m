function [ddmu_h, ddvar_h] = grads_VariationalPrs_MultiOutputGP(m,Kmats,q_sqrt,q_diag);
% function to compute gradients of 
% mu_h = C* mu_k + b
% var_h = C^2 var_k

[ddmu_k,ddvar_k] = grads_VariationalPrs_posteriorGP(m,Kmats,q_sqrt,q_diag);

for k = 1:m.dx
    % gradients with respect to q_mu{k} -- first cell entry
    ddmu_h{1,k} = mtimesx(m.prs.C(:,k),permute(ddmu_k{1,k},[4 2 1 3]));
    
    % gradients with respect to q_sqrt{k} -- second cell entry
    ddvar_h{1,k} = mtimesx(m.prs.C(:,k).^2,permute(ddvar_k{1,k},[4 2 1 3]));
    
    % gradients with respect to q_diag{k} -- third cell entry
    ddvar_h{2,k} = mtimesx(m.prs.C(:,k).^2,permute(ddvar_k{2,k},[4 2 1 3]));
end