function [ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_posteriorGP(m,Kmats,q_mu,q_sigma,R,numprsPerK);

M = size(q_mu{1},3); % either number of trials or conditions
% assert(M == size(Kmats.Kzz{1},3))

if nargin < 6
    for k = 1:m.dx
        numprsPerK(k) = m.kerns{k}.numhprs;
    end
end

for k = 1:m.dx
    
    Kzzi = Kmats.Kzzi{k};
    Ak = mtimesx(Kzzi,q_mu{k});
    
    dBhh = -mtimesx(mtimesx(permute(Kzzi,[1 2 4 3]),Kmats.dKzzhprs{k}),permute(Kzzi,[1 2 4 3]));
 
    mm1= permute(mtimesx(permute(Ak,[4 1 2 3]),permute(Kmats.dKtzhprs{k},[2 1 3 4])),[2 3 4 1]);

    mm2 = mtimesx(permute(q_mu{k},[1 2 4 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
    
    mm22 = mtimesx(reshape(mm2,[],R,1,M),'T',reshape(permute(dBhh,[2 1 5 3 4]),size(q_mu{k},1)^2,1,numprsPerK(k),[]));
    
    ddk_mu_hprs{k} = mm1 + permute(mm22,[1 3 4 2]);
    
    if nargout > 1
        
        mm4 = mtimesx(Kzzi,Kmats.Ktz{k},'T');
        mm3 = mtimesx(q_sigma{k},mm4);
        mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
        mm5 = reshape(mm5,[],1,R,M);
        mm6 = permute(reshape(dBhh,size(q_mu{k},1)^2,numprsPerK(k),[]),[2 1 4 3]); 
        mm7 = mtimesx(Kzzi,mm3);
        
        ttm1 = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.dKtzhprs{k},[2 3 1 4])),[3 2 4 1]) ...
            + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
            - 2*permute(mtimesx(permute(Kmats.dKtzhprs{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
            - permute(mtimesx(permute(Kmats.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBhh,[1 2 4 3]),Kmats.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
        
        ddk_sig_hprs{k} = bsxfun(@plus,permute(Kmats.dKtt{k},[1 3 2]),ttm1);
    end
end
