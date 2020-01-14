function m = Mstep_Update_Iterative_svGPFA(m)

% get current parameters
prs0 = [m.prs.C(:);m.prs.b];

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = BuildKernelMatrices(m,m.tt,m.Z,current_hprs,0); % get current Kernel matrices   

% predict posterior means and variances
[mu_k, var_k] = predict_posteriorGP(Kmats,m.q_mu,m.q_sigma);

% make objective function
fun = @(prs) Mstep_Objective(m,prs,mu_k,var_k);
% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('M step Grad Check:\n')
    DerivCheck(fun,prs0);
end

% minimize
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.Mstep;
prs = minFunc(fun,prs0,optimopts);

% update model parameters in structure
m.prs.C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
m.prs.b = prs(m.dy*m.dx + 1 : end);