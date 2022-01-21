function [obj,grad] = Mstep_Objective(m,prs,mu_k,var_k)

C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
b = prs(m.dy*m.dx + 1 : end);

mu_h = bsxfun(@plus,mtimesx(mu_k, C'), b');
var_h = mtimesx(var_k,(C.^2)');

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h);
gradElik = m.EMfunctions.gradLik_modelPrs(m,C,b,mu_k,var_k);

obj = - Elik; % negative free energy, KLd is constant wrt model parameters
grad = - gradElik; % gradients
