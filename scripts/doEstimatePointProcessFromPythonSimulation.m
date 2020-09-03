
%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
addpath(genpath('../src'))
addpath(genpath('~/dev/research/programs/src/matlab/iniconfig'))

pEstNumber=11120982; % test
% pEstNumber=41576915; % 3 latents, 5 trials, 100 neurons, 20 EM iterations
% pEstNumber=21161869; % 3 latents, 5 trials, 100 neurons, 50 EM iterations
% pEstNumber=49508230; % 3 latents, 5 trials, 100 neurons, 50 EM iterations
% pEstNumber=89669968; % 2 latents, 1 trial, 1 neuron , 50 EM iterations
pythonDataFilenamePattern = '../../pythonCode/scripts/results/%08d_estimationDataForMatlab.mat';

pythonDataFilename = sprintf(pythonDataFilenamePattern, pEstNumber);
pythonData = load(pythonDataFilename);

% control variables
dx = pythonData.nLatents;
dy = pythonData.nNeurons;
ntr = pythonData.nTrials;

% trials lengths
trLen = pythonData.trialsLengths;

% spikes
Y = cell(ntr,1);
for r=0:ntr-1
    Y{r+1,1} = cell(dy,1);
    for n=0:dy-1
        spikes = pythonData.(sprintf('spikesTimes_%d_%d', r, n));
        Y{r+1,1}{n+1,1} = reshape(spikes, length(spikes), 1);
    end
end

indPointsLocsGramMatrixEpsilon=pythonData.indPointsLocsKMSRegEpsilon;

% embedding params
noisyPRS.C = pythonData.C0;
noisyPRS.b = pythonData.d0;

% inducing points
Z = cell(dx,1);
for ii = 0:dx-1
    Zii = pythonData.(sprintf('indPointsLocs0_%d', ii));
    Z{ii+1} = zeros(size(Zii,2),1,ntr);
    for jj = 0:ntr-1
        Z{ii+1}(:,1,jj+1) = Zii(jj+1,:);
    end
end

% kernels
kerns = {};
for ii = 0:dx-1
    kernelType = pythonData.(sprintf('kernelType_%d', ii));
    kernelsParams0 = pythonData.(sprintf('kernelsParams0_%d', ii))';
    switch kernelType
        case 'PeriodicKernel'
            kerns{ii+1} = buildKernel('Periodic', kernelsParams0);
        case 'ExponentialQuadraticKernel'
            kerns{ii+1} = buildKernel('RBF', kernelsParams0);
        otherwise
            error(sprintf('kernelType %s not recognized', kernelType))
    end
end

% quadrature points
% legQuadPoints = pythonData.legQuadPoints;
% legQuadWeights = pythonData.legQuadWeights;

% end new code

%% set up fitting structure
% nZ = zeros(nLatents);
% for k=1:nLatents
%     nZ(k) = 10; % number of inducing points for each latent
% nZ(2) = 11;
% nZ(3) = 22;
% nZ(1) = 40; % number of inducing points for each latent
% nZ(2) = 41;
% nZ(3) = 42;
% nZ(1) = 5; % number of inducing points for each latent
% nZ(2) = 5;
% nZ(3) = 6;
Nmax = 500;
dt = max(trLen)/Nmax;

% set up kernels
% kern1 = buildKernel('Periodic',[1.5;1/2.5]);
% kern2 = buildKernel('Periodic',[1.2;1/2.5]);
% kern3 = buildKernel('RBF',1);
% kern1 = buildKernel('Periodic',[3.5;1/0.5]);
% kern2 = buildKernel('Periodic',[0.2;1/3.5]);
% kern3 = buildKernel('RBF',.5);
% kerns = {kern1, kern2,kern3};

% set up list of inducing point locations
% Z = cell(dx,1);
% for ii = 1:dx
%     for jj = 1:ntr
%         Z{ii}(:,:,jj) = linspace(dt,trLen(jj),nZ(ii))';
%     end
% end

%% initialise model structure
options.parallel = 0;
options.verbose = 1;

options.maxiter.EM = pythonData.emMaxIter;
options.maxiter.Estep = pythonData.eStepMaxIter;
options.maxiter.Mstep = pythonData.mStepEmbeddingMaxIter;
options.maxiter.hyperMstep = pythonData.mStepKernelsMaxIter;
options.maxiter.inducingPointMstep = pythonData.mStepIndPointsMaxIter;

options.nbatch = ntr;
options.nquad = length(pythonData.('legQuadPoints'));

m = InitialiseModel_svGPFA('PointProcess',@exponential,Y,trLen,kerns,Z,noisyPRS,options);
m.epsilon = indPointsLocsGramMatrixEpsilon; % value of diagonal added to kernel inversion for stability

% q_mu, q_sqrt, q_diag
q_mu = cell(1,dx);
q_sqrt = cell(1,dx);
q_diag = cell(1,dx);
for ii=0:dx-1
    nIndPointsk = size(Z{ii+1}, 1);
    q_mu{ii+1} = zeros(nIndPointsk, 1, ntr);
    q_mu{ii+1}(:,1,:) = pythonData.(sprintf('qMu0_%d', ii))';
    q_sqrt{ii+1} = zeros(nIndPointsk, 1, ntr);
    q_sqrt{ii+1}(:,1,:) = pythonData.(sprintf('qSVec0_%d', ii))';
    q_diag{ii+1} = zeros(nIndPointsk, 1, ntr);
    q_diag{ii+1}(:,1,:) = pythonData.(sprintf('qSDiag0_%d', ii))';
end
q_sigma = get_full_from_lowplusdiag(m,q_sqrt,q_diag);
m.q_mu = q_mu;
m.q_sqrt = q_sqrt;
m.q_diag = q_diag;
m.q_sigma = q_sigma;

% ttQuad, wwQuad
ttQuad = zeros(length(pythonData.('legQuadPoints')), 1, ntr);
ttQuad(:,1,:) = pythonData.('legQuadPoints')';
wwQuad(:,1,:) = pythonData.('legQuadWeights')';
m.ttQuad = ttQuad;
m.wwQuad = wwQuad;

%% set extra options and fit model
% m.opts.maxiter.EM = 50; % maximum number of iterations to run
m.opts.fixed.Z = 0; % set to 1 to hold certain parameters values fixed
m.opts.fixed.hprs = 0;
m.opts.nbatch = ntr; % number of trials to use for hyperparameter update

m = variationalEM(m);

%% predict latents and MultiOutput GP
ngtest = 2000;
testTimes = linspace(0,max(trLen),ngtest)';
pred = predictNew_svGPFA(m,testTimes);

% trueLatents = {};
% for nn = 1:ntr
%     for ii = 1:dx
%         trueLatents{nn, ii} = fs{ii,nn}(testTimes);
%     end
% end

lowerBound = m.FreeEnergy;
elapsedTime = m.elapsedTime;
meanEstimatedLatents = pred.latents.mean;
varEstimatedLatents = pred.latents.variance;

estimationParamsINI = IniConfig();
estimationParamsINI.AddSections({'data'});
estimationParamsINI.AddKeys('data', 'pEstNumber', pEstNumber);

exit = false;
while ~exit
    rNum = randi([0, 10^8-1], 1);
    estimationResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', rNum);
    estimationParamsFilename = sprintf('results/%08d-pointProcessEstimationParams.ini', rNum);
    if ~isfile(estimationResFilename) & ~isfile(estimationParamsFilename) 
        save(estimationResFilename, 'lowerBound', 'elapsedTime', 'testTimes', 'meanEstimatedLatents', 'varEstimatedLatents', 'm');
        estimationParamsINI.WriteFile(estimationParamsFilename);
        exit = true;
    end
end

%% plot latents for a given trial
% nn = 2;
% figure;
% for ii = 1:dx
%     subplot(dx,1,ii);plot(testTimes, fs{ii,nn}(testTimes), 'k', 'Linewidth', 1.5);
%     hold on; plot(testTimes, pred.latents.mean(:,ii,nn), 'Linewidth', 1.5);
%     errorbarFill(testTimes, pred.latents.mean(:,ii,nn), sqrt(pred.latents.variance(:,ii,nn)));
%     hold on; plot(m.Z{ii}(:,:,nn),min(pred.latents.mean(:,ii,nn))*ones(size(m.Z{ii}(:,:,nn))),'r.','markersize',12)
%     box off;
%     xlim([0 max(trLen)])
%     if ii ~= dx
%        set(gca,'Xtick',[]) 
%     end
%     set(gca,'TickDir','out')
%     ylabel(sprintf('x_%d',ii))
% end
% xlabel('time')
% legend('true','estimated')

% %% plot all estimated vs true log-rates / Gaussian means
% latents_nn = cell2mat(cellfun(@(x)feval(x,testTimes),fs(:,nn),'uni',0)')';
% figure;
% hold on;plot(pred.multiOutputGP.mean(:,:,nn)',prs.C*latents_nn + prs.b,':','Linewidth',1.8);
% hold on;
% plot(linspace(-5,5,100),linspace(-5,5,100),'k')
% xlabel('estimated mean')
% ylabel('true mean')
% title(sprintf('corr = %1.3f',corr(vec(pred.multiOutputGP.mean(:,:,nn)'),vec(prs.C*latents_nn + prs.b))))
% %% plot lower bound vs iteration number and vs elapsed time
% figure;
% subplot(1,2,1);
% plot(m.FreeEnergy(:,1))
% xlabel('Iteration number');
% ylabel('Free Energy');
% subplot(1,2,2);
% plot(m.elapsedTime(:,1), m.FreeEnergy(:,1))
% xlabel('Elapsed Time (sec)');
% ylabel('Free Energy');
