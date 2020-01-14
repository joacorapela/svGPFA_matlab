function kldiv = KL_divergence(S0inv,mu1,S1)
% KL divergence between N(0,S0) and N(mu1,S1)
% supports 3D array input.
ESS = S1 + mtimesx(mu1,mu1,'T');

kldiv = 0;
for nn = 1:size(mu1,3)
    aux1 = logdet(S0inv(:,:,nn));
    aux2 = logdet(S1(:,:,nn));
    aux3 = vec(S0inv(:,:,nn))'*vec(ESS(:,:,nn));
    aux4 = size(ESS,1);
    kldiv_nn =  0.5*(-logdet(S0inv(:,:,nn)) - logdet(S1(:,:,nn)) + vec(S0inv(:,:,nn))'*vec(ESS(:,:,nn))...
        - size(ESS,1));
    
    if ~isreal(kldiv_nn)
        error('complex value encountered')
    end
    
    kldiv = kldiv + kldiv_nn;
    
end

