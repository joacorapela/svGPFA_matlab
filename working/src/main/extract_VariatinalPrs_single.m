function [q_mu_new,q_sigma_new] = extract_VariatinalPrs_single(q_mu,q_sigma,trEval);

dx = length(q_mu);
q_mu_new = cell(dx,1);
q_sigma_new = cell(dx,1);
for k =1:dx
    q_mu_new{k} = q_mu{k}(:,:,trEval);
    q_sigma_new{k} = q_sigma{k}(:,:,trEval);
end