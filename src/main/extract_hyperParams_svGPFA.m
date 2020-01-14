function hprs = extract_hyperParams_svGPFA(m);

hprs = cell2mat(cellfun(@(c)c.hprs, m.kerns,'uni',0)');
