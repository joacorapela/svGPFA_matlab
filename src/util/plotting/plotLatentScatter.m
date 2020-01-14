function [interTrialMean,interTrialStd,maxEventTime] = plotLatentScatter(mu_fs,times,trialLengths,nrows,markerColor);
% function to plot the scatter/across-trial variance of the inferred
% latents
% input:
% mu_fs {Ntr} with [T x dx] -- inferred latents evaluated on 1ms grid
% times                     -- times of event to align to 
% trialLengths [Ntr]        -- length of trial in ms
%
% Lea Duncker, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntr = size(mu_fs,1);
dx = size(mu_fs{1},2);

mu_fs_aligned = alignLatents(mu_fs,times,trialLengths);

LatentTensor = cell2mat(permute(mu_fs_aligned,[3 2 1]));
interTrialMean = mean(LatentTensor,3);
interTrialStd = std(LatentTensor,[],3);

ncols = ceil(dx/nrows);
maxEventTime = times(times== max(times));
distFromEvent = trialLengths - times;
maxDistFromEvent = distFromEvent(distFromEvent== max(distFromEvent));
Tmax = maxDistFromEvent(1) + maxEventTime(1);

% tt_alg = distFromEvent-Tmax+1:distFromEvent;

figure;
for jj = 1:dx
    subplot(nrows,ncols,jj);hold on; plot(interTrialMean(:,jj),'color',[.1 .1 .1],'linewidth',1);
    plot(maxEventTime,interTrialMean(maxEventTime,jj),'.','markersize',20,'color',markerColor)
    xlim([0,Tmax])
    errorbarFill(find(~isnan(interTrialMean(:,jj))),interTrialMean(~isnan(interTrialMean(:,jj)),jj),interTrialStd(~isnan(interTrialMean(:,jj)),jj) ...
       ,.5*[.1 .1 1] ,'EdgeColor',[0.3 0.3 0.3],'FaceAlpha', 0.03);
    str = sprintf('$\\mathbf{x}_{%d}$',jj);
   title(str,'interpreter','latex','fontsize',18);
end
