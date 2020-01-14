function grad_kl = grad_variationalPrs_KL_divergence(m,Kmats,q_mu,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,trEval);

if nargin < 9
   trEval = 1:m.ntr;
   saveIdx = 1:m.ntr;
elseif length(trEval) == 1
   saveIdx(trEval) = 1; 
else
    saveIdx(trEval) = trEval;
end

for k = 1:m.dx
    q_sigma{k} =  mtimesx(q_sqrt{k},q_sqrt{k},'T') + diag3D(q_diag{k}.^2);
    
    grad_kl(idx{k},:) =  squeeze(mtimesx(Kmats.Kzzi{k},q_mu{k}));
    
    mk1 = mtimesx(Kmats.Kzzi{k},q_sqrt{k});
    mk2 = mtimesx(Kmats.Kzzi{k},diag3D(q_diag{k}));
    
    for nn = trEval
        grad_kl(idx_sig{k},saveIdx(nn)) =   vec( - q_sigma{k}(:,:,saveIdx(nn))\q_sqrt{k}(:,:,saveIdx(nn)) + mk1(:,:,saveIdx(nn)));
        grad_kl(idx_sigdiag{k},saveIdx(nn)) =  diag( - q_sigma{k}(:,:,saveIdx(nn))\diag(q_diag{k}(:,:,saveIdx(nn))) + mk2(:,:,saveIdx(nn)));
    end
end


grad_kl = grad_kl(:);