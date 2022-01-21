function [ddk_mu_in,ddk_sig_in] = grads_inducingPoints_posteriorGP(m,Kmats,q_mu,q_sigma,R);

% trEval is either the trial index or 1:m.ntr to update all trials
% simultaneously
% include helper indices to keep track of changing structure depending on
% parallel or sequential updating

M = size(q_mu{1},3);

for k = 1:m.dx
    
    q_mu_k = q_mu{k};
    q_sigma_k = q_sigma{k};
    Kzzi = Kmats.Kzzi{k};
    Ak = mtimesx(Kzzi,q_mu_k);
    
    dBzz = - mtimesx(mtimesx(permute(Kzzi,[1 2 4 3]),Kmats.dKzzin{k}),permute(Kzzi,[1 2 4 3])); % d/dZ Kzzi
    
    mm1 = permute(mtimesx(permute(Ak,[4 1 2 3]),permute(Kmats.dKtzin{k},[2 1 3 4])),[2 3 4 1]);
    
    mm2 = mtimesx(permute(q_mu_k,[1 2 4 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
    mm22 = mtimesx(reshape(mm2,[],R,1,M),'T',reshape(permute(dBzz,[2 1 5 3 4]),[],1,m.numZ(k),M));
    
    ddk_mu_in{k} = mm1 + squeeze(mm22);
    
    if nargout > 1
        
        mm4 =  mtimesx(Kzzi,Kmats.Ktz{k},'T');
        mm3 = mtimesx(q_sigma_k,mm4);
        mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
        mm5 = reshape(mm5,[],1,R,M);
        mm6 = permute(reshape(dBzz,[],m.numZ(k),M),[2 1 4 3]);
        mm7 = mtimesx(Kzzi,mm3);
        
        ddk_sig_in{k} = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.dKtzin{k},[2 3 1 4])),[3 2 4 1]) ...
            + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
            - 2*permute(mtimesx(permute(Kmats.dKtzin{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
            - permute(mtimesx(permute(Kmats.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBzz,[1 2 4 3]),Kmats.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
        
    end
    
end
