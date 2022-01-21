function pred = LeaveOneNeuronOutPrediction(m,dat,T,tt);
% LeaveOneNeuronOutPredictionError -- function to compute leave one neuron
% out prediction error on test data
% 
% input
% m     - fitted model structure
% dat   - {M} cell array with data from test trials {Neurons} or {Neur x T}
% tt    - time grid to use for computing firing rates, row vector
%
% output
% pred    - array with predicted firing rates for each held out neuron for
%           each test trial [Neurons x Time x Trials]
%
% Lea Duncker 2017 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract things from model
dy = m.dy;
Z = m.Z;
Z = cellfun(@(x) x(x < T), Z,'uni',0);
kerns = m.kerns;
ntr = max(size(dat));
pred = zeros(dy,length(tt),ntr);

for nn = 1:dy 
    
    fprintf('.')
    idx = ones(dy,1);
    idx(nn) = 0;
    idx = logical(idx);
    % extract data for all neurons but nn-th one
    Y = cellfun(@(x) x(idx,:),dat,'uni',0);
    prs.C = m.prs.C(idx,:);
    prs.b = m.prs.b(idx);
    
    options.parallel = 0;
    options.verbose = 0;
        
    switch m.lik

        case 'PointProcess'
            
            options.nquad = m.opts.nquad;
            
            mk = InitialiseModel_svGPFA('PointProcess',Y,T,kerns,Z,prs,options);
            mk.opts.fixed.Z = 1;
            mk.opts.fixed.prs = 1;
            mk.opts.fixed.hprs = 1;
            
        case 'Poisson'
           
            mk = InitialiseModel_svGPFA('Poisson',Y,T,kerns,Z,prs,options);
            mk.opts.fixed.Z = 1;
            mk.opts.fixed.prs = 1;
            mk.opts.fixed.hprs = 1;
            
    end
    % fit model with one neuron left out
    mk = svGPFA(mk);
    
    % get prediction for held-out neuron
    Kmats = KernelMatrices_prediction_svGPFA(mk,tt);
    mu_k = predict_latentGPs_svGPFA(mk,Kmats);
    
    % project outward from latent predictor using full C,b
    mu_h_n = bsxfun(@plus,mtimesx(mu_k,m.prs.C(nn,:)'), m.prs.b(nn)');
 
    % extract predictions for held out neuron
    pred(nn,:,:) = permute(mu_h_n,[2 1 3]);
end

pred = exp(pred); % map through non-linearity to get positive rates 

fprintf('\n')
