function [YSpikes,YRates,prs,rates,fs,dx,dy,ntr,trLen,tSpikes,tRates] = generate_toy_data(dy,ntr,binWidthSpikes,binWidthRates)
% dy: number of neurons
% ntr: number of trials

% Y: spike times
% prs: structure containing initial parameter values
% rates: spike rates
% fs: latents
% dx: number of latents
% dy: number of neurons
% ntr: number of trials
% trLen: trial length

dx = 3; % number of latents 
T = repmat(20,ntr,1); % maximum time for each trial
% T = [19 20];

trLen = T;
% T = [18 17 20 20 19];
fq1 = 2.5; % frequency of oscillation
fq2 = 2.5;

% build latent functions
fs = cell(dx,ntr);
for ii = 1:ntr 
    fs{1,ii}  = @(t)  0.8*exp(-2*(sin(pi*abs(t)*fq1).^2)/5).*sin(2*pi*fq1*t);
    fs{2,ii}  = @(t)  0.5*cos(2*pi*fq2*t);
    fs{3,ii}  = @(t)  0.7*exp(-0.5*(t-5).^2/8).*cos(2*pi*.12*t)  + 0.8*exp(-0.5*(t-10).^2/12).*sin(2*pi*t*0.1 + 1.5);
end

%% simulate spike train and Gaussian traces
% model parameters
prs.C = 0.4*bsxfun(@times,randn(dy,dx),[1 1.2 1.3]);
prs.b = -0.1*ones(dy,1);
% nonlinearity
nonlin = @(x) exp(x);
% grids for plotting
for i = 1:dy
    for m = 1:ntr 
        rates{i,m} = @(tt) exp(prs.C(i,:)*cell2mat(cellfun(@(x) x(tt), fs(:,m),'uni',0)) + prs.b(i));
    end
end

% Begin modified by Joaco 09/03/19

ngridSpikes = T/binWidthSpikes;
tSpikes = cell(ntr,1);
for nn = 1:ntr 
    tSpikes{nn} = linspace(0,T(nn),ngridSpikes(nn)); % time grid for sampling spike train
end
% generate spikes
sps = simulate_spikes_from_rates(rates,T,ngridSpikes);
YSpikes = cell(ntr,1);
for nn = 1:ntr; YSpikes{nn} = sps(:,nn);end % this is just to keep things consitent with other inputs
% print total number of spikes across neurons and plot rates and raster
fprintf('total number of spikes across all neurons: %d \n',sum(cellfun(@(x)size(x,1),sps)))


ngridRates = T/binWidthRates;
tRates = cell(ntr,1);
for nn = 1:ntr 
    tRates{nn} = linspace(0,T(nn),ngridRates(nn)); % time grid for sampling spike train
end
YRates = cell(ntr,1);
for nn = 1:ntr
    YRates{nn} = zeros(dy,ngridRates(nn));
    for neuronID = 1:dy
        YRates{nn}(neuronID,:) = rates{neuronID,nn}(tRates{nn});
    end
end

% keyboard
% End modified by Joaco 09/03/19
