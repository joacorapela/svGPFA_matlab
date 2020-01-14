function grad = gradLik_ModelPrs_PointProcess(m,C,b,mu_k_Quad,var_k_Quad,mu_k_Spikes,var_k_Spikes,varargin);

if nargin < 8
    mu_hQuad = bsxfun(@plus,mtimesx(mu_k_Quad,C'), b');
    var_hQuad = mtimesx(var_k_Quad,(C.^2)');
    if ~strcmp(func2str(m.link),'exponential')
        for nn = 1:m.ntr
            mu_hObs{nn} = bsxfun(@plus,sum(bsxfun(@times,mu_k_Spikes{nn},C(m.index{nn},:)),2), b(m.index{nn}));
            var_hObs{nn} = sum(bsxfun(@times,var_k_Spikes{nn},C(m.index{nn},:).^2),2);
        end
    end
else
    mu_hQuad = varargin{1};
    var_hQuad = varargin{2};
    if  ~strcmp(func2str(m.link),'exponential')
        mu_hObs = varargin{3};
        var_hObs = varargin{4};
    end
end

if strcmp(func2str(m.link),'exponential') % use closed form expectations
    intval = exp(mu_hQuad + 0.5*var_hQuad); % T x N x m.ntr
    
    quadIntval = bsxfun(@times,m.wwQuad,intval);
    gradCtt1 = mtimesx(quadIntval,'T',mu_k_Quad);
    gradCtt2 = permute(mtimesx(permute(mtimesx(permute(var_k_Quad,[1 4 2 3]),...
        permute(C,[3 1 2 4])),[5 1 2 3 4]),permute(quadIntval,[1 5 2 4 3])),[3 4 5 1 2]);
    
    grad_C = gradCtt1 + gradCtt2;
    
    grad_b = permute(mtimesx(m.wwQuad,'T',intval),[2 3 1]);
    for nn =  1:m.ntr
        mm1 = m.spikecellID{nn};
        grad_C2(:,:,nn) = mm1'*mu_k_Spikes{nn};
        grad_b2(:,nn) = sum(mm1,1)';
    end
    
else % use  quadrature
    
    [~,dlinkvalQuad] = m.link(sqrt(2 * var_hQuad) .* permute(m.xxHerm,[2 3 4 1]) + mu_hQuad); % should be N x T x R x Nq (check these dims)
        
    dintval =  m.wwQuad .* permute(mtimesx(m.wwHerm,'T',permute(dlinkvalQuad,[4 1 2 3])),[2 3 4 1]); % derivative of expectation term E(g'(x))
    dsigintval = m.wwQuad .* permute(mtimesx(m.wwHerm,'T',permute( dlinkvalQuad.*(sqrt(2)/2 ./ sqrt(var_hQuad) .* permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);
    
    gradCtt1 = mtimesx(dintval,'T',mu_k_Quad);
    
    gradCtt2 = 2*C.*mtimesx(dsigintval,'T',var_k_Quad);
    
    grad_C = gradCtt1 + gradCtt2;
    
    grad_b = permute(sum(dintval,1),[2 3 1]);
    
    for nn =  1:m.ntr
        
        [linkvalObs,dlinkvalObs] = m.link(sqrt(2 * var_hObs{nn}) .* m.xxHerm' + mu_hObs{nn}); % should be Nspikes x Nquad_link
        loglinkQuad1 = (dlinkvalObs./linkvalObs)*m.wwHerm;
        loglinkQuad2 = (sqrt(2)/2./sqrt(var_hObs{nn}).*(dlinkvalObs./linkvalObs).*m.xxHerm')*m.wwHerm;
        
        mm1 = m.spikecellID{nn};
        grad_C2(:,:,nn) = mm1'*(mu_k_Spikes{nn}.*loglinkQuad1) + 2*C.*(mm1'*(var_k_Spikes{nn}.*loglinkQuad2)) ;
        grad_b2(:,nn) = mm1'*loglinkQuad1;
    end
    
end

grad_poi1 = sum([reshape(grad_C,[],m.ntr); grad_b],2);
grad_poi2 = sum([reshape(grad_C2,[],m.ntr); grad_b2],2);

grad = -grad_poi1 + full(grad_poi2);


