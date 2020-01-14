function [mu_k,var_k] = predict_posteriorGP(Kmats,q_mu,q_sigma)
% function to predict expected posterior GP from Kernel matrices and
% variational means and covariances
%
% factors inside the sum of Eq. (5) in Duncker and Sahani, 2018
% q_mu is a cell of length K (number of latents)
% q_mu{k} is a 3D matrix for trial k: numberOfInducingPoints x 1 x number of trials
% q_sigma is a cell of length K (number of latents)
% q_sigma{k} is a 3D matrix for trial k: numberOfInducingPoints x numberOfInducingPoints x number of trials
% mu_k \in nQuad x nLatent x nTrial
% sigma_k \in nQuad x nLatent x nTrial

for k = 1:length(q_mu)
    
    q_mu_k = q_mu{k}; % now 3D over trials
    q_sigma_k = q_sigma{k};
    Kzzi = Kmats.Kzzi{k};
    Ak = mtimesx(Kzzi,q_mu_k);
    
    Bkf = mtimesx(Kzzi,Kmats.Ktz{k},'T');
    mm1f = mtimesx(q_sigma_k - Kmats.Kzz{k},Bkf);
    mu_k(:,k,:) = mtimesx(Kmats.Ktz{k},Ak);

    % original code by Lea Duncker
    % var_k(:,k,:) = bsxfun(@plus,Kmats.Ktt(:,k,:),sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));

    % updated by Joaco 07/08/2019 to save one permute
    var_k(:,k,:) = bsxfun(@plus,Kmats.Ktt(:,k,:),permute(sum(bsxfun(@times,Bkf,mm1f),1),[2,1,3]));

end
