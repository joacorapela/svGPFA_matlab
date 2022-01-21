function [mu_k,var_k] = predict_posteriorGP_fromSpikes(KmatsSpikes,Kzzi,q_mu,q_sigma,trEval)
% function to predict expected posterior GP from Kernel matrices and
% variational means and covariances

if size(q_mu{1},3) == 1 % single parameter case
    qqEval = ones(length(trEval),1);
else
    qqEval = 1:length(trEval);
end

% precompute stuff that is shared over nn indices later
for k = 1:length(q_mu)
    Ak{k} = mtimesx(Kzzi{k},q_mu{k});
end

% compute Numspikes x K predictions for each trial 
for nn = 1:length(trEval)
    for k = 1:length(q_mu)   
        qn = qqEval(nn);
        Bkf = mtimesx(Kzzi{k}(:,:,qn),KmatsSpikes.Ktz{k,nn},'T');
        mm1f = mtimesx(q_sigma{k}(:,:,qn) - pinv(Kzzi{k}(:,:,qn)),Bkf);
        mu_k{nn}(:,k) = KmatsSpikes.Ktz{k,nn}*Ak{k}(:,:,qn);
        if nargout > 1
            var_k{nn}(:,k) = bsxfun(@plus,KmatsSpikes.Ktt{k,nn}, sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));
        end        
    end
end