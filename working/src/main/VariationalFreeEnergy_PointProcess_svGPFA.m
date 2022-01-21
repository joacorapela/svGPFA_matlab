function fe = VariationalFreeEnergy_PointProcess_svGPFA(m)

% compute kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams

% get current Kernel matrices for quadrature 
Kmats_Quad = BuildKernelMatrices(m,m.ttQuad,m.Z,current_hprs,0);

% get current Kernel matrices evaluated at observed spike data
Kmats_Spikes = BuildKernelMatrices_fromSpikes(m,m.Z,current_hprs,0);

% get multi output prediction for quad and from spikes
[mu_h_Quad,var_h_Quad] = m.EMfunctions.predict_MultiOutputGP(Kmats_Quad,m.q_mu,m.q_sigma,m.prs.C,m.prs.b);
[mu_h_Spikes,var_h_Spikes] = m.EMfunctions.predict_MultiOutputGP_fromSpikes(Kmats_Spikes,Kmats_Quad.Kzzi,Kmats_Quad.Kzz,m.q_mu,m.q_sigma,m.prs.C,m.prs.b,1:m.ntr,m.index);

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h_Quad,var_h_Quad,mu_h_Spikes,var_h_Spikes);

% get KL divergence 
KLd = build_KL_divergence(m,Kmats_Quad,m.q_mu,m.q_sigma);

fe = Elik - KLd;
