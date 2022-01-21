function [mu_h_n_Quad,mu_h_n_Spikes] = get_single_rates_grouped_PointProcess(prs,mu_k_Mean_Quad,mu_k_Group_Quad,mu_k_Noise_Quad,...
    mu_k_Mean_Spikes,mu_k_Group_Spikes,mu_k_Noise_Spikes,idx);

nPred = find(~idx); % neuron which was left out

if isstruct(prs.C)
    C_Mean = prs.C.Mean;
    C_Group = prs.C.Group;
    C_Noise = prs.C.Noise;
else
    C_Mean = prs.C;
    C_Group = prs.C;
    C_Noise = prs.C;
end
    
% get multi output for quadrature
mu_h_Mean_Quad = mtimesx(mu_k_Mean_Quad, C_Mean(nPred,:)');

mu_h_Group_Quad = mtimesx(mu_k_Group_Quad, C_Group(nPred,:)');

mu_h_Noise_Quad = mtimesx(mu_k_Noise_Quad, C_Noise(nPred,:)');

mu_h_n_Quad = bsxfun(@plus,mu_h_Group_Quad + mu_h_Noise_Quad + mu_h_Mean_Quad,prs.b(nPred)');


% get multi output for from-spike predictions
for nn = 1:length(mu_k_Noise_Spikes);
    if ~isempty(mu_k_Noise_Spikes{nn})
        mu_h_n_Spikes{nn} = bsxfun(@plus,sum(bsxfun(@times,mu_k_Group_Spikes{nn},C_Group(nPred,:)),2) ...
            + sum(bsxfun(@times,mu_k_Noise_Spikes{nn},C_Noise(nPred,:)),2)...
            + sum(bsxfun(@times,mu_k_Mean_Spikes{nn},C_Mean(nPred,:)),2), prs.b(nPred));
    else
        mu_h_n_Spikes{nn} = [];
    end
end
