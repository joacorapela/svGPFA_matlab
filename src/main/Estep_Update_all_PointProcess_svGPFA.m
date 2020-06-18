function prs = Estep_Update_all_PointProcess_svGPFA(m)

qmu0 = cell2mat(m.q_mu(:));
qsqrt0 = cell2mat(m.q_sqrt(:));
qdiag0 = cell2mat(m.q_diag(:));

prs0 = [qmu0;qsqrt0;qdiag0];
prs0 = prs0(:);

% extract current hyperparameters
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams

% get current Kernel matrices for quadrature 
Kmats_Quad = BuildKernelMatrices(m,m.ttQuad,m.Z,current_hprs,0);
% get current Kernel matrices evaluated at observed spike data
Kmats_Spikes = BuildKernelMatrices_fromSpikes(m,m.Z,current_hprs,0,1:m.ntr);

% make objective function

fun = @(prs) Estep_Objective_PointProcess_svGPFA(m,prs,Kmats_Quad,Kmats_Spikes);

% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('E step Grad Check:\n')
    DerivCheck(fun,prs0);
end
% run optimizer
optimopts = optimset('Gradobj','on','display', 'iter');
optimopts.MaxIter = m.opts.maxiter.Estep;

[prs, nLowerBound, exitfag, output] = minFunc(fun,prs0,optimopts);

% begin debug

q_mu = m.q_mu;
q_sqrt = m.q_sqrt;
q_diag = m.q_diag;
C = m.prs.C;
b = m.prs.b;
index = m.index;
ttQuad = m.ttQuad;
wwQuad = m.wwQuad;
xxHerm = m.xxHerm;
wwHerm = m.wwHerm;
Z = m.Z;
Y = m.Y;
epsilon = m.epsilon;
hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';

kernelNames = {};
for k=1:length(m.kerns)
    kernelNames{k} = func2str(m.kerns{k}.K);
end

maxIter = optimopts.MaxIter;

filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/code/ci/data/Estep_Update_all_PointProcess_svGPFA.mat';
save(filename, 'epsilon', 'q_mu', 'q_sqrt', 'q_diag', 'C', 'b', 'index', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z', 'Y', 'kernelNames', 'hprs', 'maxIter', 'prs', 'nLowerBound', 'exitfag', 'output');

keyboard

% end debug
