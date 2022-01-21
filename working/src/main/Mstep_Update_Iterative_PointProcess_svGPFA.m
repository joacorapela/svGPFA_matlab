function m = Mstep_Update_Iterative_PointProcess_svGPFA(m)

C0 = m.prs.C;
b0 = m.prs.b;

% get current parameters
prs0 = [m.prs.C(:);m.prs.b];

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats_Quad = BuildKernelMatrices(m,m.ttQuad,m.Z,current_hprs,0); % get current Kernel matrices  
Kmats_Spikes = BuildKernelMatrices_fromSpikes(m,m.Z,current_hprs,0); % get current Kernel matrices evaluated at observed spike data

% predict posterior means and variances
[mu_k_Quad, var_k_Quad] = predict_posteriorGP(Kmats_Quad,m.q_mu,m.q_sigma);
[mu_k_Spikes, var_k_Spikes] = predict_posteriorGP_fromSpikes(Kmats_Spikes,Kmats_Quad.Kzzi,m.q_mu,m.q_sigma,1:m.ntr);

% make objective function
fun = @(prs) Mstep_Objective_PointProcess(m,prs,mu_k_Quad,var_k_Quad,mu_k_Spikes,var_k_Spikes);

% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('M step Grad Check:\n')
    DerivCheck(fun,prs0);
end
% minimize
% optimopts = optimset('Gradobj','on','display', 'iter');
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.Mstep;
[prs, nLowerBound, exitfag, output] = minFunc(fun,prs0,optimopts);

% update model parameters in structure
m.prs.C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
m.prs.b = prs(m.dy*m.dx + 1 : end);

% begin debug

% warning('Debug code is running on Mstep_Update_Iterative_PointProcess_svGPFA.m');
% 
% q_mu = m.q_mu;
% q_sqrt = m.q_sqrt;
% q_diag = m.q_diag;
% C = m.prs.C;
% b = m.prs.b;
% index = m.index;
% ttQuad = m.ttQuad;
% wwQuad = m.wwQuad;
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% Z = m.Z;
% Y = m.Y;
% %
% epsilon = m.epsilon;
% 
% hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';
% 
% kernelNames = {};
% for k=1:length(m.kerns)
%     kernelNames{k} = func2str(m.kerns{k}.K);
% end
% 
% maxIter = optimopts.MaxIter;
% 
% prsC = m.prs.C;
% prsb = m.prs.b;
% 
% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/Mstep_Update_Iterative_PointProcess_svGPFA.mat';
% save(filename, 'epsilon', 'q_mu', 'q_sqrt', 'q_diag', 'C0', 'b0', 'index', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z', 'Y', 'hprs', 'kernelNames', 'maxIter', 'prsC', 'prsb', 'nLowerBound', 'exitfag', 'output');
% 
% keyboard

% end debug
