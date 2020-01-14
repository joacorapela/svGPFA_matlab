function grad = gradLik_hprs_Poisson_svGPFA(m,Kmats,idxhprs,ntr,mu_h,var_h,q_mu,q_sigma);

M = length(ntr);
nhprs = sum(cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0)));
grad_lik_poi1 = zeros(nhprs,M);
grad_lik_poi2 = zeros(nhprs,M);

mask = permute(repmat(m.mask(:,ntr),[1 1 m.dy]),[1 3 2]); % check this
R = max(m.trLen);

% gradients of posterior means and variances wrt hyperparams
[ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_posteriorGP(m,Kmats,q_mu,q_sigma,R);

    
if strcmp(func2str(m.link),'exponential') % use closed form expectations
    
    intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
    intval(mask) = 0;
    
    for k = 1:m.dx
        
        logExp = bsxfun(@times,permute(ddk_mu_hprs{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_hprs{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
        
        grad_lik_poi1(idxhprs{k},:) = permute(m.BinWidth*sum(reshape(bsxfun(@times,permute(intval,[1 2 4 3]),logExp),[],m.kerns{k}.numhprs,M),1),[2 3 1]);
        
        grad_lik_poi2(idxhprs{k},:) = permute(mtimesx(mtimesx(m.prs.C(:,k)',m.Y(:,:,ntr)),ddk_mu_hprs{k}),[2 3 1]);
    end
    
else % use Gauss-Hermite quadrature
    
    [linkval,dlinkval] = m.link(sqrt(2 * var_h) .* permute(m.xxHerm,[2 3 4 1]) + mu_h); % should be N x T x R x Nq (check these dims)
    dlinkval(mask) = 0;
        
    for k = 1:m.dx
        
        innerExp = bsxfun(@times,permute(ddk_mu_hprs{k},[1 5 2 3 4]),m.prs.C(:,k)') ...
            + sqrt(2)/2./permute(sqrt(var_h),[1 2 4 3]).*bsxfun(@times,permute(ddk_sig_hprs{k},[1 5 2 3 4]),m.prs.C(:,k).^2')...
            .* permute(m.xxHerm,[2 3 5 4 1]);
              
        grad_lik_poi1(idxhprs{k},:) = permute(m.BinWidth*sum(reshape(sum(permute(dlinkval,[1 2 5 3 4]).*innerExp...
            .*permute(m.wwHerm,[5 4 2 3 1]),5),[],m.kerns{k}.numhprs,M),1),[2 3 1]);
        
        grad_lik_poi2(idxhprs{k},:) = permute(sum(reshape(permute(m.Y(:,:,ntr),[2 1 4 3])...
            .*sum(permute(dlinkval./linkval,[1 2 5 3 4]).*innerExp...
            .*permute(m.wwHerm,[5 4 2 3 1]),5),[],m.kerns{k}.numhprs,M),1),[2 3 1]);
        
    end
    
end

grad = sum(-grad_lik_poi1 + grad_lik_poi2,2);
% grad = sum(grad_lik_poi2,2);
