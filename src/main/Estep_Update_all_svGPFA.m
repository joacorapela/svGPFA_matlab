function prs = Estep_Update_all_svGPFA(m)

qmu0 = cell2mat(m.q_mu(:));
qsqrt0 = cell2mat(m.q_sqrt(:));
qdiag0 = cell2mat(m.q_diag(:));

prs0 = [qmu0;qsqrt0;qdiag0];
prs0 = prs0(:);

% extract current hyperparameters
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams

% build kernel matrices
Kmats = BuildKernelMatrices(m,m.tt,m.Z,current_hprs,0);

% make objective function
fun = @(prs) Estep_Objective_svGPFA(m,prs,Kmats);

% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('E step Grad Check:\n')
    DerivCheck(fun,prs0);
end
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

[prs, nLowerBound, exitfag, output] = minFunc(fun,prs0,optimopts);

% begin debug

% q_mu = m.q_mu;
% q_sqrt = m.q_sqrt;
% q_diag = m.q_diag;
% C = m.prs.C;
% b = m.prs.b;
% tt = m.tt;
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% Z = m.Z;
% Y = m.Y;
% BinWidth = m.BinWidth;
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
% filename = '~/dev/research/gatsby/svGPFA/code/ci/data/Estep_Update_all_svGPFA.mat';
% save(filename, 'q_mu', 'q_sqrt', 'q_diag', 'tt', 'xxHerm', 'wwHerm', 'Z', 'Y', 'C', 'b', 'kernelNames', 'hprs', 'BinWidth', 'maxIter', 'prs', 'nLowerBound', 'exitfag', 'output');
% 
% keyboard

% end debug

