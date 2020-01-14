function kl_grad = grad_hprs_KL_divergence(m,Kmats,idxhprs,q_mu,q_sigma);
% gradient of KL diveregence wrt hyperparameters, computed for all trials
% simultanously.

numhprsPerK = cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0));

R = size(q_mu{1},3);
assert(R == size(Kmats.Kzzi{1},3));

for k = 1:m.dx
    kl_grad_hprs = [];
    
    Eqq = q_sigma{k} + mtimesx(q_mu{k},q_mu{k},'T');
    dKzzi = -0.5*pinv3D(Kmats.Kzzi{k}) + 0.5*(Eqq);
    
    Kzzi = Kmats.Kzzi{k};
    for hh = 1: numhprsPerK(k)
        kl_grad_hprs = [kl_grad_hprs;...
            mtimesx(reshape(dKzzi,[],1,R),'T',...
            reshape(-mtimesx(Kzzi,mtimesx(permute(Kmats.dKzzhprs{k}(:,:,hh,:),[1 2 4 3]),Kzzi)),[],1,R))];
    end
    
    kl_grad(idxhprs{k},:) = permute(kl_grad_hprs,[1 3 2]);
end

kl_grad = sum(kl_grad,2);