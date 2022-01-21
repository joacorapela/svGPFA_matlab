function [ddmu_k,ddvar_k] = grads_VariationalPrs_posteriorGP(m,Kmats,q_sqrt,q_diag);

for k = 1:m.dx

    Ktz = Kmats.Ktz{k};
    Kzz = Kmats.Kzz{k};
    Kzzi = pinv3D(Kzz);
    
    Ak = mtimesx(Kzzi,Ktz,'T');
    
    % save gradient of  mu_k wrt q_mu
    ddmu_k{1,k} = Ak;

    % Kzzi * Kzt * Ktz * Kzzi
    B = bsxfun(@times,permute(Ak,[1 4 2 3]),permute(Ak,[4 1 2 3]));
    % gradients wrt to variance parameters
    ddvar_k{1,k} = 2 * permute(mtimesx(B,permute(q_sqrt{k},[1 2 4 3])),[1 3 4 2]);
%     ddvar_k{2,k} = 2 * mtimesx(permute(diag3D(q_diag{k}),[1 2 4 3]),B);
    
    ddvar_k{2,k} = 2 * permute(reshape(diag3D(reshape(bsxfun(@times,permute(q_diag{k},[1 2 4 3]),B),...
        m.numZ(k),m.numZ(k),[])),m.numZ(k),1,[],m.ntr),[1 3 4 2]);
    
end
