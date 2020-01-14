function prs = inducingPointMstep_single(m,ntr)

% extract hyperparameters and inducing points for each latent process
zz = cell2mat(m.Z);
prs0 = zz(:,:,ntr);

% make objective function
fun = @(prs) m.EMfunctions.InducingPointMstep_Objective(m,prs,ntr);

% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('inducing point grad check:\n')
    DerivCheck(fun,prs0);
end
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

prs = minFunc(fun,prs0,optimopts);

end