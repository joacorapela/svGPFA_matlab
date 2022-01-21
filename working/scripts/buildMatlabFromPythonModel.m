
function m = buildMatlabFomPythonModel(pythonModelFilename)
    pythonModel = load(pythonModelFilename);

    % control variables
    dx = pythonModel.nLatents;
    dy = pythonModel.nNeurons;
    ntr = pythonModel.nTrials;
    
    % trials lengths
    trLen = pythonModel.trialsLengths;
    
    % spikes
    Y = cell(ntr,1);
    for r=0:ntr-1
        Y{r+1,1} = cell(dy,1);
        for n=0:dy-1
            spikes = pythonModel.(sprintf('spikesTimes_%d_%d', r, n));
            Y{r+1,1}{n+1,1} = reshape(spikes, length(spikes), 1);
        end
    end
    
    indPointsLocsGramMatrixEpsilon=pythonModel.indPointsLocsKMSRegEpsilon;
    
    % embedding params
    noisyPRS.C = pythonModel.C;
    noisyPRS.b = pythonModel.d;
    
    % inducing points
    Z = cell(dx,1);
    for ii = 0:dx-1
        Zii = pythonModel.(sprintf('indPointsLocs_%d', ii));
        Z{ii+1} = zeros(size(Zii,2),1,ntr);
        for jj = 0:ntr-1
            Z{ii+1}(:,1,jj+1) = Zii(jj+1,:);
        end
    end
    
    % kernels
    kerns = {};
    for ii = 0:dx-1
        kernelType = pythonModel.(sprintf('kernelType_%d', ii));
        kernelsParams = pythonModel.(sprintf('kernelsParams_%d', ii))';
        switch kernelType
            case 'PeriodicKernel'
                kerns{ii+1} = buildKernel('Periodic', kernelsParams);
            case 'ExponentialQuadraticKernel'
                kerns{ii+1} = buildKernel('RBF', kernelsParams);
            otherwise
                error(sprintf('kernelType %s not recognized', kernelType))
        end
    end
    
    Nmax = 500;
    dt = max(trLen)/Nmax;
    
    %% initialise model structure
    options.parallel = 0;
    options.verbose = 1;
    
    options.maxiter.EM = pythonModel.emMaxIter;
    options.maxiter.Estep = pythonModel.eStepMaxIter;
    options.maxiter.Mstep = pythonModel.mStepEmbeddingMaxIter;
    options.maxiter.hyperMstep = pythonModel.mStepKernelsMaxIter;
    options.maxiter.inducingPointMstep = pythonModel.mStepIndPointsMaxIter;
    
    options.nbatch = ntr;
    options.nquad = length(pythonModel.('legQuadPoints'));
    
    m = InitialiseModel_svGPFA('PointProcess',@exponential,Y,trLen,kerns,Z,noisyPRS,options);
    m.epsilon = indPointsLocsGramMatrixEpsilon; % value of diagonal added to kernel inversion for stability
    
    % q_mu, q_sqrt, q_diag
    q_mu = cell(1,dx);
    q_sqrt = cell(1,dx);
    q_diag = cell(1,dx);
    for ii=0:dx-1
        nIndPointsk = size(Z{ii+1}, 1);
        q_mu{ii+1} = zeros(nIndPointsk, 1, ntr);
        q_mu{ii+1}(:,1,:) = pythonModel.(sprintf('qMu_%d', ii))';
        q_sqrt{ii+1} = zeros(nIndPointsk, 1, ntr);
        q_sqrt{ii+1}(:,1,:) = pythonModel.(sprintf('qSVec_%d', ii))';
        q_diag{ii+1} = zeros(nIndPointsk, 1, ntr);
        q_diag{ii+1}(:,1,:) = pythonModel.(sprintf('qSDiag_%d', ii))';
    end
    q_sigma = get_full_from_lowplusdiag(m,q_sqrt,q_diag);
    m.q_mu = q_mu;
    m.q_sqrt = q_sqrt;
    m.q_diag = q_diag;
    m.q_sigma = q_sigma;
    
    % ttQuad, wwQuad
    ttQuad = zeros(length(pythonModel.('legQuadPoints')), 1, ntr);
    ttQuad(:,1,:) = pythonModel.('legQuadPoints')';
    wwQuad(:,1,:) = pythonModel.('legQuadWeights')';
    m.ttQuad = ttQuad;
    m.wwQuad = wwQuad;
    
    %% set extra options and fit model
    m.opts.fixed.Z = 0; % set to 1 to hold certain parameters values fixed
    m.opts.fixed.hprs = 0;
    m.opts.nbatch = ntr; % number of trials to use for hyperparameter update
    
end
