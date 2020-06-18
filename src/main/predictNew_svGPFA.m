function pred = predictNew_svGPFA(m, testTimes);

current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = BuildKernelMatrices(m,testTimes,m.Z,current_hprs,0); % get current Kernel matrices   

[mu_k, var_k] = predict_posteriorGP(Kmats,m.q_mu,m.q_sigma);

mu_h = bsxfun(@plus,mtimesx(mu_k,m.prs.C'), m.prs.b');
var_h = mtimesx(var_k,(m.prs.C.^2)');

pred.latents.mean = mu_k;
pred.latents.variance = var_k;
pred.multiOutputGP.mean = mu_h;
pred.multiOutputGP.variance = var_h;

% start debug

% q_mu = m.q_mu;
% q_sqrt = m.q_sqrt;
% q_diag = m.q_diag;
% C = m.prs.C;
% b = m.prs.b;
% ttQuad = m.ttQuad;
% wwQuad = m.wwQuad;
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% Z = m.Z;
% Y = m.Y;
% hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';
% kernelNames = {};
% for k=1:length(m.kerns)
%     kernelNames{k} = func2str(m.kerns{k}.K);
% end

% end debug

% start debug

% muK = mu_k;
% varK = var_k;
% muH = mu_h;
% varH = var_h;
% 
% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/predictNew_svGPFA.mat'; 
% save(filename, 'q_mu', 'q_sqrt', 'q_diag', 'C', 'b', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z', 'Y', 'hprs', 'kernelNames', 'testTimes', 'muK', 'varK', 'muH', 'varH');
% 
% keyboard

% end debug

