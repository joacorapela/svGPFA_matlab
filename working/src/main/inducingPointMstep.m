function m = inducingPointMstep(m)

if ~m.opts.fixed.Z
    m = inducingPointMstep_Update(m); 
end