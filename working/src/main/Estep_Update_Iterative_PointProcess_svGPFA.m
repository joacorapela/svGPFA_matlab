function m = Estep_Update_Iterative_PointProcess_svGPFA(m)

if m.ntr == 1
    % single trial case, no need to parallelise
    prs = Estep_Update_single_PointProcess_svGPFA(m,1);
    
elseif m.opts.parallel % optimise variational parameters for each trial in parallel
    
    parfor (nn = 1:m.ntr,m.opts.numWorkers)
        prs{nn} = Estep_Update_single_PointProcess_svGPFA(m,nn);
    end
    
else % optimise variational parameters over all trials at once
    
    prs = Estep_Update_all_PointProcess_svGPFA(m);
    
end

% update model structure
m = update_variationalPrs_svGPFA(m,prs);
