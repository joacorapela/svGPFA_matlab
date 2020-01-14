function [ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_multiOutputGP_fromSpikes(m,KmatsSpike,Kzzi,dKzzhprs,q_mu,q_sigma,trEval,C,numhprsPerK);

if nargin < 8
    for k = 1:m.dx
       numhprsPerK(k) = m.kerns{k}.numhprs; 
    end 
end
R = length(trEval);
    
if size(Kzzi{1},3) == 1
    qqEval = ones(R,1);
else
    qqEval = 1:R;
end

for k = 1:m.dx
    
    Ak = mtimesx(Kzzi{k},q_mu{k});

    dBhh = - mtimesx(mtimesx(permute(Kzzi{k},[1 2 4 3]),dKzzhprs{k}),permute(Kzzi{k},[1 2 4 3])); % d/dhprs Kzzi
                

    for nn = 1:R
        ddObshprs1 = permute(mtimesx(Ak(:,:,nn),'T',permute(KmatsSpike.dKtzhprs{k,nn},[2 1 3])),[2 3 1]);
        ddObshprs2 = permute(mtimesx(mtimesx(KmatsSpike.Ktz{k,nn},dBhh(:,:,:,qqEval(nn))),q_mu{k}(:,:,nn)),[1 3 2]);
        ggn{nn} = C(m.index{trEval(nn)},k)'*(ddObshprs1 + ddObshprs2);
    end
    
    ddk_mu_hprs{k} = ggn;    
    
    if nargout > 1
                       
        mm6 = permute(reshape(dBhh,size(q_mu{k},1)^2,numhprsPerK(k),[]),[1 2 4 3]); % vec(dBhh)
        mm7 = mtimesx(mtimesx(Kzzi{k},q_sigma{k}),Kzzi{k}); % Kzzi S Kzzi
                
        % dsig/dhprs = dKtt + 2 Tr[Kzzi dKzz Kzzi S Kzzi Psi2
        
        for nn = 1:R
            Nspks = length(m.index{trEval(nn)});
            
            Kzttz = bsxfun(@times,permute(KmatsSpike.Ktz{k,nn},[2 3 1]),permute(KmatsSpike.Ktz{k,nn},[3 2 1]));

            dKzttzhprs = 2 * reshape(bsxfun(@times,permute(KmatsSpike.dKtzhprs{k,nn},[2 3 1]),permute(KmatsSpike.Ktz{k,nn},[3 2 1])),[],Nspks,numhprsPerK(k));

            % Tr(dKzzi S Kzzi Kzttz) + Tr(Kzzi S dKzzi Kzttz) + Tr(Kzzi S Kzzi dKzttz)
            ttm1 =  permute(2*mtimesx(reshape(mtimesx(dBhh(:,:,:,qqEval(nn)),permute(mtimesx(q_sigma{k}(:,:,nn),Kzzi{k}(:,:,qqEval(nn))),[1 2 4 3])),[],1,numhprsPerK(k),1),'T',...
                permute(reshape(Kzttz,[],1,Nspks,1),[1 3 2 4])),[2 3 4 1]) ...
                + permute(mtimesx(reshape(mm7(:,:,nn),[],1),'T',dKzttzhprs),[2 3 4 1]);
            
            % Tr(Kzzi dKzttz) + Tr(dKzzi Kzttz)
            ttm2 = permute(mtimesx(reshape(Kzzi{k}(:,:,qqEval(nn)),size(q_mu{k},1)^2,1,1,[]),'T',dKzttzhprs),[2 3 4 1]) + permute(mtimesx(mm6(:,:,:,qqEval(nn)),'T',reshape(Kzttz,[],1,Nspks)),[3 1 4 2]);
            sigggn{nn} =  C(m.index{trEval(nn)},k).^2'*bsxfun(@plus,KmatsSpike.dKtt{k,nn},ttm1 - ttm2);
        end
        ddk_sig_hprs{k} = sigggn;
        
        
    end
end