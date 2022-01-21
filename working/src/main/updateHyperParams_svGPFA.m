function m = updateHyperParams_svGPFA(m,prs)

hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'uni',0)'));
istrthprs = [1; hprsidx(1:end-1)+1];
iendhprs = hprsidx;

for kk = 1:m.dx
    m.kerns{kk}.hprs = prs(istrthprs(kk):iendhprs(kk));
end