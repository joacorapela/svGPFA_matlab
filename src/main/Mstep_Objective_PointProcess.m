function [obj,grad] = Mstep_Objective_PointProcess(m,prs,mu_k_Quad,var_k_Quad,mu_k_Spikes,var_k_Spikes)

C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
b = prs(m.dy*m.dx + 1 : end);

% get multi output for quadrature
mu_h_Quad = bsxfun(@plus,mtimesx(mu_k_Quad, C'), b');
var_h_Quad = mtimesx(var_k_Quad,(C.^2)');

% get multi output for from-spike predictions
for nn = 1:m.ntr
    mu_h_Spikes{nn} = bsxfun(@plus,sum(bsxfun(@times,mu_k_Spikes{nn},C(m.index{nn},:)),2), b(m.index{nn}));
    var_h_Spikes{nn} = sum(bsxfun(@times,var_k_Spikes{nn},C(m.index{nn},:).^2),2);
end

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h_Quad,var_h_Quad,mu_h_Spikes,var_h_Spikes);
gradElik = m.EMfunctions.gradLik_modelPrs(m,C,b,mu_k_Quad,var_k_Quad,mu_k_Spikes,var_k_Spikes,mu_h_Quad,var_h_Quad,mu_h_Spikes,var_h_Spikes);

obj = - Elik; % negative free energy, KLd is constant wrt model parameters
grad = - gradElik; % gradients
