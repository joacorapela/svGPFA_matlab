function m = hyperMstep(m)
% function to update hyperparameters
if ~m.opts.fixed.hprs
    m = hyperMstep_Update(m);
end
