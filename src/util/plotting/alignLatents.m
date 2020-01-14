function mu_fs_aligned = alignLatents(mu_fs,times,trialLengths);
% function to align inferred latents to behavioural event
% latents
% input:
% mu_fs {Ntr} with [T x dx] -- inferred latents evaluated on 1ms grid
% times                     -- times of event to align to 
% trialLengths [Ntr]        -- length of trial in ms
% output:
% mu_fs_aligned {Ntr} with [T x dx] -- aligned latents 
%
% Lea Duncker, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntr = size(mu_fs,3);
dx = size(mu_fs,2);

maxEventTime = times(times== max(times));
distFromEvent = trialLengths(:) - times(:);

maxDistFromEvent = distFromEvent(distFromEvent== max(distFromEvent));

Tmax = maxDistFromEvent(1) + maxEventTime(1);

mu_fs_aligned = NaN(Tmax,dx,ntr);

for nn = 1:ntr
    % make empty matrix to fill
    % align latents to event on given trial with NaN padding
    mu_fs_aligned(maxEventTime - times(nn) + 1 : maxEventTime - times(nn) + trialLengths(nn),:,nn) ...
        = mu_fs(1:trialLengths(nn),:,nn);
end