function q_sigma = get_full_from_lowplusdiag(m,q_sqrt,q_diag,varargin)

if nargin < 4
    numZ = m.numZ;
else
    numZ = varargin{1};
end
    
R = size(q_sqrt{1},3); % number of trials/conditions on third axis
for ii = 1:m.dx
    qq =  reshape(q_sqrt{ii},numZ(ii),m.opts.varRnk(ii),R);
    dd = diag3D(q_diag{ii}.^2);
    q_sigma{ii} = mtimesx(qq,qq,'T') + dd;
end

% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/get_full_from_lowplusdiag.mat';
% save(filename, 'q_sqrt', 'q_diag', 'q_sigma')
% 
