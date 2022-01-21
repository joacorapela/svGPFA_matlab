function [ddk_mu_in,ddk_sig_in] = grads_inducingPoints_multiOutputGP_fromSpikes(m,KmatsSpike,Kzzi,dKzzin,q_mu,q_sigma,trEval,C);

% trEval is either the trial index or 1:m.ntr to update all trials
% simultaneously
% include helper indices to keep track of changing structure depending on
% parallel or sequential updating

R = length(trEval);

for k = 1:m.dx
    
    q_mu_k = q_mu{k};
    Ak = mtimesx(Kzzi{k},q_mu_k);
    
    dBzz = -mtimesx(Kzzi{k},reshape(permute(reshape(mtimesx(Kzzi{k},...
        reshape(dKzzin{k},m.numZ(k),[],R)),...
        m.numZ(k),m.numZ(k),[]),[2 1 3]),...
        m.numZ(k),m.numZ(k)^2,[]));
    
    dBzz = reshape(dBzz,m.numZ(k),m.numZ(k),[],R);
    
    for nn = 1:R
        ddObsin1 = permute(mtimesx(Ak(:,:,nn),'T',permute(KmatsSpike.dKtzin{k,nn},[2 1 3])),[2 3 1]);
        ddObsin2 = permute(mtimesx(mtimesx(KmatsSpike.Ktz{k,nn},dBzz(:,:,:,nn)),q_mu_k(:,:,nn)),[1 3 2]);
        ggn{nn} = C(m.index{trEval(nn)},k)'*(ddObsin1 + ddObsin2);
    end
    
    ddk_mu_in{k} = ggn;

    if nargout > 1
        
        error('gradient not implemented yet for Point Process likelihood')
        
    end
    
end
      
