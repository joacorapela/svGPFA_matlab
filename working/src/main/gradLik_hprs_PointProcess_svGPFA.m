function grad = gradLik_hprs_PointProcess_svGPFA(m,KmatsQuad,KmatsSpike,idxhprs,mu_hQuad,var_hQuad,q_mu,q_sigma,trEval);

M = length(trEval);
nhprs = sum(cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0)));
grad_lik_pp1 = zeros(nhprs,M);
grad_lik_pp2 = zeros(nhprs,M);

intval = exp(mu_hQuad + 0.5*var_hQuad); % T x N x length(trEval)

R = m.opts.nquad;

[ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_posteriorGP(m,KmatsQuad,q_mu,q_sigma,R);
ddk_mu_hprsObs = grads_hprs_multiOutputGP_fromSpikes(m,KmatsSpike,KmatsQuad.Kzzi,KmatsQuad.dKzzhprs,q_mu,q_sigma,trEval,m.prs.C);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_hprs{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_hprs{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    grad_lik_pp1(idxhprs{k},:) = permute(sum(mtimesx(permute(m.wwQuad(:,:,trEval),[2 1 4 3]),bsxfun(@times,permute(intval,[1 2 4 3]),logExp)),2),[3 4 2 1]);
    grad_lik_pp2(idxhprs{k},:) = cell2mat(ddk_mu_hprsObs{k}')';
    
end
    
grad = sum(-grad_lik_pp1 + grad_lik_pp2,2);


