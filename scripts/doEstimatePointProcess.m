%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
addpath(genpath('../src'))
addpath(genpath('~/dev/research/programs/src/matlab/iniconfig'))

completelyRandomEmbedding = false;
embeddingSNR = 2;
% simulationNumber=95041940; %05 seconds long
% simulationNumber=46476422; %10 seconds long
simulationNumber=85331586; %20 seconds long
% simulationNumber=98591910; %40 seconds long
indPointsLocsGramMatrixEpsilon=1e-2;

simFilename = sprintf('results/%08d-pointProcessSimulation.mat', simulationNumber);
load(simFilename);

noiseEmbeddingVar = mean(abs(prs.C(:)))/embeddingSNR;
if completelyRandomEmbedding
    noisyPRS.C = randn(size(prs.C))*noiseEmbeddingVar;
    noisyPRS.b = randn(size(prs.b))*noiseEmbeddingVar;
else
    noisyPRS.C = prs.C + randn(size(prs.C))*noiseEmbeddingVar;
    noisyPRS.b = prs.b + randn(size(prs.b))*noiseEmbeddingVar;
end

%% set up fitting structure
% nZ(1) = 10; % number of inducing points for each latent
% nZ(2) = 11;
% nZ(3) = 22;
nZ(1) = 40; % number of inducing points for each latent
nZ(2) = 41;
nZ(3) = 42;
Nmax = 500;
dt = max(trLen)/Nmax;

% set up kernels
kern1 = buildKernel('Periodic',[1.5;1/2.5]);
kern2 = buildKernel('Periodic',[1.2;1/2.5]);
kern3 = buildKernel('RBF',1);
% kern1 = buildKernel('Periodic',[3.5;1/0.5]);
% kern2 = buildKernel('Periodic',[0.2;1/3.5]);
% kern3 = buildKernel('RBF',.5);
kerns = {kern1, kern2,kern3};

% set up list of inducing point locations
Z = cell(dx,1);
for ii = 1:dx
    for jj = 1:ntr
        Z{ii}(:,:,jj) = linspace(dt,trLen(jj),nZ(ii))';
    end
end

%% initialise model structure
options.parallel = 0;
options.verbose = 1;
options.maxiter.hyperMstep = 10;
options.nbatch = ntr;
options.nquad = 200;
% m = InitialiseModel_svGPFA('PointProcess',@exponential,Y,trLen,kerns,Z,prs,options);
% modified by Joaco 09/09/19
% blas_lib = '/usr/lib/x86_64-linux-gnu/libblas.so';
% mex('-DDEFINEUNIX','mtimesx.c',blas_lib);
m = InitialiseModel_svGPFA('PointProcess',@exponential,Y,trLen,kerns,Z,noisyPRS,options);
m.epsilon = indPointsLocsGramMatrixEpsilon; % value of diagonal added to kernel inversion for stability

estimationParamsINI = IniConfig();
estimationParamsINI.AddSections({'data', 'control_variables', 'kernels_params', 'latents_params', 'indPoints_params'});
estimationParamsINI.AddKeys('data', 'simulationNumber', simulationNumber);
estimationParamsINI.AddKeys('control_variables', 'epsilon', indPointsLocsGramMatrixEpsilon);
for i=1:length(kerns)
    kernel = kerns{i};
    estimationParamsINI.AddKeys('kernels_params', sprintf('k%dType', i), func2str(kernel.K));
    estimationParamsINI.AddKeys('kernels_params', sprintf('k%dHPRS', i), kernel.hprs);
end

estimationParamsINI.AddKeys('embedding_params', 'completelyRandomEmbedding', int8(completelyRandomEmbedding));
estimationParamsINI.AddKeys('embedding_params', 'snr', embeddingSNR);
estimationParamsINI.AddKeys('indPoints_params', 'numberIndPoints', nZ);

%% set extra options and fit model
m.opts.maxiter.EM = 50; % maximum number of iterations to run
m.opts.fixed.Z = 0; % set to 1 to hold certain parameters values fixed
m.opts.fixed.hprs = 0;
m.opts.nbatch = ntr; % number of trials to use for hyperparameter update

% start debug

q_mu0 = m.q_mu;
q_sqrt0 = m.q_sqrt;
q_diag0 = m.q_diag;
C0 = m.prs.C;
b0 = m.prs.b;
ttQuad = m.ttQuad;
wwQuad = m.wwQuad;
xxHerm = m.xxHerm;
wwHerm = m.wwHerm;
Z0 = m.Z;
YNonStacked = Y;
m.YNonStacked = YNonStacked;
Y = m.Y;
index = m.index;
hprs0 = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';
kernelNames = {};
for k=1:length(m.kerns)
    kernelNames{k} = func2str(m.kerns{k}.K);
end

% end debug

m = variationalEM(m);

%% predict latents and MultiOutput GP
ngtest = 2000;
testTimes = linspace(0,max(trLen),ngtest)';
pred = predictNew_svGPFA(m,testTimes);

trueLatents = {};
for nn = 1:ntr
    for ii = 1:dx
        trueLatents{nn, ii} = fs{ii,nn}(testTimes);
    end
end

lowerBound = m.FreeEnergy;
elapsedTime = m.elapsedTime;
meanEstimatedLatents = pred.latents.mean;
varEstimatedLatents = pred.latents.variance;

exit = false;
while ~exit
    rNum = randi([0, 10^8-1], 1);
    initialConditionsFilename = sprintf('results/%08d-pointProcessInitialConditions.mat', rNum);
    estimationResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', rNum);
    estimationParamsFilename = sprintf('results/%08d-pointProcessEstimationParams.ini', rNum);
    if ~isfile(initialConditionsFilename) & ~isfile(estimationResFilename) & ~isfile(estimationParamsFilename) 
        save(initialConditionsFilename, 'q_mu0', 'q_sqrt0', 'q_diag0', 'C0', 'b0', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z0', 'Y', 'index', 'hprs0', 'kernelNames');
        save(estimationResFilename, 'lowerBound', 'elapsedTime', 'testTimes', 'meanEstimatedLatents', 'varEstimatedLatents', 'm');
        estimationParamsINI.WriteFile(estimationParamsFilename);
        exit = true;
    end
end

%% plot latents for a given trial
nn = 2;
figure;
for ii = 1:dx
    subplot(dx,1,ii);plot(testTimes, fs{ii,nn}(testTimes), 'k', 'Linewidth', 1.5);
    hold on; plot(testTimes, pred.latents.mean(:,ii,nn), 'Linewidth', 1.5);
    errorbarFill(testTimes, pred.latents.mean(:,ii,nn), sqrt(pred.latents.variance(:,ii,nn)));
    hold on; plot(m.Z{ii}(:,:,nn),min(pred.latents.mean(:,ii,nn))*ones(size(m.Z{ii}(:,:,nn))),'r.','markersize',12)
    box off;
    xlim([0 max(trLen)])
    if ii ~= dx
       set(gca,'Xtick',[]) 
    end
    set(gca,'TickDir','out')
    ylabel(sprintf('x_%d',ii))
end
xlabel('time')
legend('true','estimated')

%% plot all estimated vs true log-rates / Gaussian means
latents_nn = cell2mat(cellfun(@(x)feval(x,testTimes),fs(:,nn),'uni',0)')';
figure;
hold on;plot(pred.multiOutputGP.mean(:,:,nn)',prs.C*latents_nn + prs.b,':','Linewidth',1.8);
hold on;
plot(linspace(-5,5,100),linspace(-5,5,100),'k')
xlabel('estimated mean')
ylabel('true mean')
title(sprintf('corr = %1.3f',corr(vec(pred.multiOutputGP.mean(:,:,nn)'),vec(prs.C*latents_nn + prs.b))))
%% plot lower bound vs iteration number and vs elapsed time
figure;
subplot(1,2,1);
plot(m.FreeEnergy(:,1))
xlabel('Iteration number');
ylabel('Free Energy');
subplot(1,2,2);
plot(m.elapsedTime(:,1), m.FreeEnergy(:,1))
xlabel('Elapsed Time (sec)');
ylabel('Free Energy');
