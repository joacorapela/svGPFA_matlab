function m = initialise_VariationalPrs_svGPFA(m);

% set level of initial variance
init_variance = 0.01;

% set initial values for variational mean and low-rank plus diagonal
% representation of variational covariance
for ii = 1:m.dx
    q_mu{ii} = zeros(m.numZ(ii),1,m.ntr);
    q_sqrt{ii} = repmat(vec(init_variance*eye(m.numZ(ii),m.opts.varRnk(ii))),[1 1 m.ntr]);
    q_diag{ii} = repmat(init_variance*ones(m.numZ(ii),1),[1 1 m.ntr]);
end

% construct variational covariance from low-rank plus diag
q_sigma = get_full_from_lowplusdiag(m,q_sqrt,q_diag);

% update model structure
m.q_mu = q_mu;
m.q_sqrt = q_sqrt;
m.q_diag = q_diag;
m.q_sigma = q_sigma;
