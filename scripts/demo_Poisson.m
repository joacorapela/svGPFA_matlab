%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
rng(2)
set_paths
%% make simulated data
dy = 50; % number of neurons
ntr = 5; % number of trials
nRatesToPlot = 3;
% generate dataset with three latents
[YSpike,prs,rates,fs,dx,dy,ntr,trLen,tt] = generate_toy_data(dy,ntr);

YRate = cell(ntr,1);
for nn=1:ntr
    YRate{nn} = zeros(dy,length(tt));
    for neuronI=1:size(YRate{nn},1)
        YRate{nn}(neuronI,:)=rates{neuronI,nn}(tt);
    end
end

% visualise firing rates
figure;
subplot(121);plotRaster(YSpike{1}); title('raster plot of spike times')
subplot(122);hold on; for i = 1:nRatesToPlot; plot(tt,YRate{1}(i,:));end
title('example firing rates')
%% set up fitting structure
nZ(1) = 10; % number of inducing points for each latent
nZ(2) = 11;
nZ(3) = 12;
Nmax = 500;
dt = max(trLen)/Nmax;

% set up kernels
kern1 = buildKernel('Periodic',[1.5;1/2.5]);
kern2 = buildKernel('Periodic',[1.2;1/2.5]);
kern3 = buildKernel('RBF',1);
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
m = InitialiseModel_svGPFA('Poisson',@exponential,YRate,trLen,kerns,Z,prs,options);

%% set extra options and fit model
m.opts.maxiter.EM = 50; % maximum number of iterations to run
m.opts.fixed.Z = 0; % set to 1 to hold certain parameters values fixed
m.opts.fixed.hprs = 0;
m.opts.nbatch = ntr; % number of trials to use for hyperparameter update
m = variationalEM(m);

%% predict latents and MultiOutput GP
ngtest = 2000;
testTimes = linspace(0,max(trLen),ngtest)';
pred = predictNew_svGPFA(m,testTimes);
%% plot latents for a given trial
nn = 2;
figure; 
for ii = 1:dx
    subplot(3,1,ii);plot(testTimes,fs{ii,nn}(testTimes),'k','Linewidth',1.5);
    hold on; plot(testTimes,pred.latents.mean(:,ii,nn),'Linewidth',1.5);
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
