function [mu_warp, var_warp] = predict_warpOnly(m,testTimes);

% extract current hyperparameters
% hprs_Mean = cellfun(@(struct)struct.hprs, m.kerns_Mean,'uni',0)'; % extract kernel hyperparams
% hprs_Group = cellfun(@(struct)struct.hprs, m.kerns_Group,'uni',0)'; % extract kernel hyperparams
% hprs_Noise = cellfun(@(struct)struct.hprs, m.kerns_Noise,'uni',0)'; % extract kernel hyperparams
hprs_Warp = m.kerns_Warp.hprs; 

% % get Kernel matrices for given trial
% Kmats_standardTime = BuildKernelMatrices_grouped_svGPFA(m,testTimes,m.Z,m.Z_mean,hprs_Mean,hprs_Group,hprs_Noise,0,'all');   

Kmats_warp =  BuildKernelMatrices_WarpOnly(m,testTimes,m.Z_Warp,hprs_Warp,0);
[mu_warp, var_warp] = predict_posteriorGP_single(Kmats_warp,testTimes,m.q_mu_Warp,m.q_sigma_Warp);
