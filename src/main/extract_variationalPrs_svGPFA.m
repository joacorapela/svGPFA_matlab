function [q_mu,q_sqrt,q_diag,q_sigma, idx, idx_sig,idx_sigdiag] = extract_variationalPrs_svGPFA(m,prs,trEval)
% function to update model structure with new parameters

q_mu = cell(m.dx,1);
q_sqrt = cell(m.dx,1);
q_diag = cell(m.dx,1);

idx = cell(m.dx,1);
idx_sig = cell(m.dx,1);
idx_sigdiag = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

num_cov = m.numZ.*m.opts.varRnk;
istrt_sig = iend(end) + [1 cumsum(num_cov(1:end-1))+1];
iend_sig  = iend(end) + cumsum(num_cov);

istrt_sigdiag = iend_sig(end) + [1 cumsum(m.numZ(1:end-1))+1];
iend_sigdiag = iend_sig(end) + cumsum(m.numZ);

for kk = 1:m.dx
    idx{kk} = istrt(kk):iend(kk);
    idx_sig{kk} = istrt_sig(kk):iend_sig(kk);
    idx_sigdiag{kk} = istrt_sigdiag(kk):iend_sigdiag(kk);
end

prs = reshape(prs,[],1,length(trEval));
for k = 1:m.dx
    q_mu{k} = prs(idx{k},:,:);
    q_sqrt{k} = reshape(prs(idx_sig{k},:,:),m.numZ(k),m.opts.varRnk(k),length(trEval));
    q_diag{k} = prs(idx_sigdiag{k},:,:);
end

q_sigma = get_full_from_lowplusdiag(m,q_sqrt,q_diag);

