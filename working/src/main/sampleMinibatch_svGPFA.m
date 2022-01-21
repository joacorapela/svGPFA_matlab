function m = sampleMinibatch_svGPFA(m)

m.minibatches = cell2mat(arrayfun(@(xx) randperm(m.ntr,m.opts.nbatch), 1:(m.opts.maxiter.hyperMstep + 5), 'uni', 0)');
