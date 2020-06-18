function lik_pp = expectedLogLik_PointProcess(m,mu_h_Quad,var_h_Quad,mu_h_Spikes,var_h_Spikes,ntr)

% var_h_Spikes is supplied for later extensions for different link
% functions
if nargin < 6
    trEval = 1:m.ntr;
else
    trEval = ntr;
end

if strcmp(func2str(m.link),'exponential') % use closed form expectations for exponential nonlinearity
% warning('Test code not using analytical calculation')
% if false
    
    intval = exp(mu_h_Quad + 0.5*var_h_Quad); % numQuad x N x length(trEval)
    log_link = cellvec(mu_h_Spikes);

else % compute Gaussian expectations with quadrature

    intval = permute(mtimesx(m.wwHerm',permute(m.link(mu_h_Quad + sqrt(2*var_h_Quad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);    
    log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0)); 

end

% first term in Eq.7 of Duncker and Sahani, 2018
% using Legendre quadature due to definite integral
lik_pp1 = sum(vec(mtimesx(m.wwQuad(:,:,trEval),'T',intval)));
% second term in Eq.7 of Duncker and Sahani, 2018
lik_pp2 = sum(log_link);
lik_pp = -lik_pp1 + lik_pp2;

% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% wwQuad = m.wwQuad;
% ttQuad = m.ttQuad;
% 
% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/expectedLogLik_PointProcess.mat';
% save(filename, 'mu_h_Quad', 'var_h_Quad', 'mu_h_Spikes', 'var_h_Spikes','xxHerm', 'wwHerm', 'ttQuad', 'wwQuad', 'lik_pp', '-v6');
