function m = Mstep(m);

if ~ m.opts.fixed.prs
    m = m.EMfunctions.Mstep_Update(m);
end