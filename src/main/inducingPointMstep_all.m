function prs = inducingPointMstep_all(m);

Z0 = m.Z;
prs0 = vec(cell2mat(m.Z));

% make objective function
fun = @(prs) m.EMfunctions.InducingPointMstep_Objective(m,prs);
if m.opts.verbose == 2 % extra level of verbosity 
    fprintf('inducing point grad check:\n')
    DerivCheck(fun,prs0);
end
% run optimizer
optimopts = optimset('Gradobj','on','display', 'iter');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

[prs, nLowerBound, exitfag, output] = minFunc(fun,prs0,optimopts);

% beging debug

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
Y = m.Y;
Zf = m.Z;
epsilon = m.epsilon;

hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';

kernelNames = {};
for k=1:length(m.kerns)
    kernelNames{k} = func2str(m.kerns{k}.K);
end

maxIter = optimopts.MaxIter;

filename = '~/dev/research/gatsby/svGPFA/code/test/data/inducingPointsMstep_all.mat';
save(filename, 'epsilon', 'q_mu', 'q_sqrt', 'q_diag', 'C', 'b', 'index', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z0', 'Zf', 'Y', 'hprs', 'kernelNames', 'maxIter', 'prs', 'nLowerBound', 'exitfag', 'output');

keyboard

% end debug

