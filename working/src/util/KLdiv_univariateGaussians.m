function kld = KLdiv_univariateGaussians(m1,m2,s1,s2);
% function to compute the KL divergence between univariate gaussians:
% KLdiv_univariateGaussians(m1,m2,s1,s2) = KL[N(m1,s1) || N(m2,s2)]
% input is expected to be tensors of arbitrary but matching dimensions,
% divergences are computed elementwise

kld = 0.5*( s1./s2 + (m2-m1).^2 ./ s2 - 1 + log(s2./s1));