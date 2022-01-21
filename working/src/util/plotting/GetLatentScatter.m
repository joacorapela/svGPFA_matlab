function [interTrialMean,interTrialStd,maxEventTime,meanEndTime] = GetLatentScatter(mu_fs,times,trialLengths,nrows);
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
ntr = size(mu_fs,3);
dx = size(mu_fs,2);

mu_fs_aligned = alignLatents(mu_fs,times,trialLengths);

interTrialMean = mean(mu_fs_aligned,3);
interTrialStd = std(mu_fs_aligned,[],3);

ncols = ceil(dx/nrows);
maxEventTime = times(times== max(times));
distFromEvent = trialLengths - times;
meanEndTime = maxEventTime + floor(mean(distFromEvent));
maxDistFromEvent = distFromEvent(distFromEvent== max(distFromEvent));
Tmax = maxDistFromEvent + maxEventTime;

