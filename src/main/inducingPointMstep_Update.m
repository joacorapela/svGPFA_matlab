function m = inducingPointMstep_Update(m)
% function to update hyperparameters of svGPFA model

R = size(m.Z{1},3);
istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

if m.opts.parallel
    parfor (nn = 1:R,m.opts.numWorkers) % runs parfor if cluster is open, otherwise sequential
        prs{nn} = inducingPointMstep_single(m,nn);
    end
    
    % update inducing point values in model
    for nn = 1:R
        for kk = 1:m.dx
            m.Z{kk}(:,:,nn) = prs{nn}(istrt(kk):iend(kk));
        end
    end
    
else
    
    prs = inducingPointMstep_all(m);
    
    % update inducing point values in model
    prs = reshape(prs,[],1, R);
    for kk = 1:m.dx
        m.Z{kk} = prs(istrt(kk):iend(kk),:,:);
    end
    
end