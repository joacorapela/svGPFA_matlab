function grad = gradLik_inducingPoints_PointProcess_svGPFA(m,KmatsQuad,KmatsSpike,idxZ,trEval,mu_h_Quad,var_h_Quad,q_mu,q_sigma);

grad_lik_pp1 = zeros(sum(m.numZ),length(trEval));
grad_lik_pp2 = zeros(sum(m.numZ),length(trEval));


intval = exp(mu_h_Quad + 0.5*var_h_Quad); % T x N x length(trEval)

R = m.opts.nquad;

[ddk_mu_in,ddk_sig_in] = grads_inducingPoints_posteriorGP(m,KmatsQuad,q_mu,q_sigma,R);
ddk_mu_inObs= grads_inducingPoints_posteriorGP_fromSpikes(m,KmatsSpike,KmatsQuad.Kzzi,KmatsQuad.dKzzin,q_mu,q_sigma,trEval);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_in{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_in{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    
    grad_lik_pp1(idxZ{k},:) = permute(sum(mtimesx(permute(m.wwQuad(:,:,trEval),[2 1 4 3]),bsxfun(@times,permute(intval,[1 2 4 3]),logExp)),2),[3 4 2 1]);   
                              
    grad_lik_pp2(idxZ{k},:) = cell2mat(ddk_mu_inObs{k}')';
    
end

grad = - grad_lik_pp1(:) + grad_lik_pp2(:);
    
