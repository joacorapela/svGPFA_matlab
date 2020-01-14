function grad = gradLik_VariationalPrs_Poisson_svGPFA(m,Kmats,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,trEval,mu_h,var_h);

R = length(trEval);
nvprs = 2*sum(m.numZ) + sum(m.numZ.*m.opts.varRnk);
grad_lik_poi1 = zeros(nvprs,R);
grad_lik_poi2 = zeros(nvprs,R);

mask = permute(repmat(m.mask(:,trEval),[1 1 m.dy]),[1 3 2]); % check this

if strcmp(func2str(m.link),'exponential') % use closed form expectations
    intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
    intval(mask) = 0;
    
    for k = 1:m.dx
        
        Ktz = Kmats.Ktz{k};
        Kzzi = Kmats.Kzzi{k};
        
        Ak = mtimesx(Kzzi,Ktz,'T');
        
        grad_lik_poi1(idx{k},:) = permute(m.BinWidth*sum(mtimesx(Ak,mtimesx(intval,m.prs.C(:,k))),2),[1 3 2]);
        grad_lik_poi2(idx{k},:) = permute(sum(mtimesx(Ak,mtimesx(m.prs.C(:,k)',m.Y(:,:,trEval)),'T'),2),[1 3 2]);
        
        Bt = m.BinWidth*mtimesx(intval,m.prs.C(:,k).^2); % T x 1 x ntr
        KztOut = bsxfun(@times,permute(Ktz,[2 4 1 3]),permute(Ktz,[4 2 1 3]));
        tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
        
        grad_lik_poi1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,q_sqrt{k})),[],1,length(trEval)),[1 3 2]);
        grad_lik_poi1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,diag3D(q_diag{k})))),[1 3 2]);
        
    end
else % use Gauss Hermite Quadrature for calculating expectations
    
    [linkval,dlinkval] = m.link(sqrt(2 * var_h) .* permute(m.xxHerm,[2 3 4 1]) + mu_h); % should be N x T x R x Nq (check these dims)
    dlinkval(mask) = 0;
    dintval =  permute(mtimesx(m.wwHerm,'T',permute(dlinkval,[4 1 2 3])),[2 3 4 1]); % derivative of expectation term E(g'(x))
%   dintval =  sum(dlinkval.*permute(m.wwHerm,[4 2 3 1]),4); % derivative of expectation term E(g'(x))

    dintval(mask) = 0;
        
    for k = 1:m.dx
        
        Ktz = Kmats.Ktz{k};
        Kzzi = Kmats.Kzzi{k};
        
        Ak = mtimesx(Kzzi,Ktz,'T');
        
        % gradient of E(g(x)) wrt q_mu
        grad_lik_poi1(idx{k},:) = permute(m.BinWidth*sum(mtimesx(Ak,mtimesx(dintval,m.prs.C(:,k))),2),[1 3 2]);
        % gradient of E(log g(x)) wrt q_mu
        grad_lik_poi2(idx{k},:) = permute(sum(mtimesx(Ak,mtimesx(m.prs.C(:,k)',...
            m.Y(:,:,trEval).*permute(sum(dlinkval./linkval.*permute(m.wwHerm,[4 2 3 1]),4),[2 1 3])),'T'),2),[1 3 2]);
                        
        Bt = m.BinWidth*mtimesx(sum(dlinkval.*(sqrt(2)./sqrt(var_h)...
            .*permute(m.xxHerm,[2 3 4 1])).*permute(m.wwHerm,[4 2 3 1]),4),m.prs.C(:,k).^2); % T x 1 x ntr
            
        KztOut = bsxfun(@times,permute(Ktz,[2 4 1 3]),permute(Ktz,[4 2 1 3]));
        tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
        
        Bt2 = mtimesx(permute(m.Y(:,:,trEval),[2 1 3]).*sum(dlinkval./linkval.*(sqrt(2)./sqrt(var_h)...
            .*permute(m.xxHerm,[2 3 4 1])).*permute(m.wwHerm,[4 2 3 1]),4),m.prs.C(:,k).^2); % T x 1 x ntr
            
        tSum2 = permute(sum(bsxfun(@times, KztOut, permute(Bt2, [4 2 1 3])),3),[1 2 4 3]);
        
        % gradient of E(g(x)) wrt q_sigma and q_diag
        grad_lik_poi1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,q_sqrt{k})),[],1,length(trEval)),[1 3 2]);
        grad_lik_poi1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,diag3D(q_diag{k})))),[1 3 2]);
        
        % gradient of E(log g(x)) wrt q_sigma and q_diag
        grad_lik_poi2(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(Kzzi,tSum2),mtimesx(Kzzi,q_sqrt{k})),[],1,length(trEval)),[1 3 2]);
        grad_lik_poi2(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(Kzzi,tSum2),mtimesx(Kzzi,diag3D(q_diag{k})))),[1 3 2]);
        
    end
end

grad = - grad_lik_poi1(:) + grad_lik_poi2(:);