function kl_grad = grad_inducingPoints_KL_divergence(m,Kmats,idxZ,q_mu, q_sigma,trEval);
% currently only for a single trial assuming Z always updates separately

kl_grad  = zeros(sum(m.numZ),length(trEval));

% KL Divergence gradients

for k = 1:m.dx
    
    kl_grad_Z = [];
    % iterate over latent processes
    
    q_mu_k = q_mu{k};
    q_sigma_k = q_sigma{k};
    
    Eqq = q_sigma_k + mtimesx(q_mu_k,q_mu_k,'T');
    dKzzi = -0.5*pinv3D(Kmats.Kzzi{k}) + 0.5*(Eqq);
    Kzzi = Kmats.Kzzi{k};
    
    for ii = 1: m.numZ(k)
        kl_grad_Z = [kl_grad_Z;...
            mtimesx(reshape(dKzzi,[],1,length(trEval)),'T',...
            reshape(-mtimesx(Kzzi,mtimesx(permute(Kmats.dKzzin{k}(:,:,ii,:),[1 2 4 3]),Kzzi)),[],1,length(trEval)))];
    end

    kl_grad(idxZ{k},:) = permute(kl_grad_Z,[1 3 2]);
    
end

kl_grad = kl_grad(:);
