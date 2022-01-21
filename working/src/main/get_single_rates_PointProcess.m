function [mu_h_n,mu_h_n_Spikes] = get_single_rates_PointProcess(prs,mu_k_Quad,mu_k_Spikes,idx);

% get output rate for n_th neuron
mu_h_n = bsxfun(@plus,mtimesx(mu_k_Quad,prs.C(nn,:)'), prs.b(nn)');

for tr = 1:m.ntr % this is for all neurons
    mu_h_Spikes{tr} = bsxfun(@plus,sum(bsxfun(@times,mu_k_Spikes{tr},prs.C(m.index{tr},:)),2), b(m.index{tr}));
end

mu_h_n_Spikes =