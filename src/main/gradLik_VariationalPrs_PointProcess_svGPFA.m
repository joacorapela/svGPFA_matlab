function grad = gradLik_VariationalPrs_PointProcess_svGPFA(m,KmatsQuad,KmatsObs,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,trEval,mu_h_Quad,var_h_Quad);

R = length(trEval);
nvprs = 2*sum(m.numZ) + sum(m.numZ.*m.opts.varRnk);
grad_lik_pp1 = zeros(nvprs,R);
grad_lik_pp2 = zeros(nvprs,R);

if strcmp(func2str(m.link),'exponential') % use closed form expectations
    
    intval = exp(mu_h_Quad + 0.5*var_h_Quad); % numQuad x N x length(trEval)
    
    for k = 1:m.dx
        
        q_sqrt_k = q_sqrt{k};
        q_diag_k = q_diag{k};
        
        AkQuad = mtimesx(KmatsQuad.Kzzi{k},KmatsQuad.Ktz{k},'T');
        
        grad_lik_pp1(idx{k},:) = mtimesx(bsxfun(@times,permute(mtimesx(intval,m.prs.C(:,k)),[2 1 3]),AkQuad),m.wwQuad(:,:,trEval));
        
        for nn = 1:R
            KtzObs = KmatsObs.Ktz{k,nn};
            AkObs = KmatsQuad.Kzzi{k}(:,:,nn)*KtzObs';
            grad_lik_pp2(idx{k},nn) = AkObs*m.prs.C(m.index{trEval(nn)},k);
        end
        
        Bt = m.wwQuad(:,:,trEval).*mtimesx(intval,(m.prs.C(:,k).^2)); % Tx1 vector
        KztOut = bsxfun(@times,permute(KmatsQuad.Ktz{k},[2 4 1 3]),permute(KmatsQuad.Ktz{k},[4 2 1 3]));
        tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
        
        grad_lik_pp1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(KmatsQuad.Kzzi{k},tSum),mtimesx(KmatsQuad.Kzzi{k},q_sqrt_k)),[],1,length(trEval)),[1 3 2]);
        grad_lik_pp1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(KmatsQuad.Kzzi{k},tSum),mtimesx(KmatsQuad.Kzzi{k},diag3D(q_diag_k)))),[1 3 2]);
        
    end
    
else
    
    [linkval,dlinkval] = m.link(sqrt(2 * var_h) .* permute(m.xxHerm,[2 3 4 1]) + mu_h); % should be N x T x R x Nq (check these dims)
    dlinkval(mask) = 0;
    dintval =  permute(mtimesx(m.wwHerm,'T',permute(dlinkval,[4 1 2 3])),[2 3 4 1]); % derivative of expectation term E(g'(x))

    for k = 1:m.dx
        
        q_sqrt_k = q_sqrt{k};
        q_diag_k = q_diag{k};
        
        AkQuad = mtimesx(KmatsQuad.Kzzi{k},KmatsQuad.Ktz{k},'T');
        
        grad_lik_pp1(idx{k},:) = mtimesx(bsxfun(@times,permute(mtimesx(dintval,m.prs.C(:,k)),[2 1 3]),AkQuad),m.wwQuad(:,:,trEval));
        
        for nn = 1:R
            KtzObs = KmatsObs.Ktz{k,nn};
            AkObs = KmatsQuad.Kzzi{k}(:,:,nn)*KtzObs';
            grad_lik_pp2(idx{k},nn) = AkObs*m.prs.C(m.index{trEval(nn)},k);
        end
        
        Bt = m.wwQuad(:,:,trEval).*mtimesx(intval,(m.prs.C(:,k).^2)); % Tx1 vector
        KztOut = bsxfun(@times,permute(KmatsQuad.Ktz{k},[2 4 1 3]),permute(KmatsQuad.Ktz{k},[4 2 1 3]));
        tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
        
        grad_lik_pp1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(KmatsQuad.Kzzi{k},tSum),mtimesx(KmatsQuad.Kzzi{k},q_sqrt_k)),[],1,length(trEval)),[1 3 2]);
        grad_lik_pp1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(KmatsQuad.Kzzi{k},tSum),mtimesx(KmatsQuad.Kzzi{k},diag3D(q_diag_k)))),[1 3 2]);
        
    end
    
end
grad = - grad_lik_pp1(:) + grad_lik_pp2(:);

