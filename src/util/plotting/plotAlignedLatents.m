function [mu_fs_aligned]=plotAlignedLatents(mu_fs,times,times2,trialLengths,nrows,markerColors,varargin);
% function to plot the scatter/across-trial variance of the inferred
% latents
% input:
% mu_fs {Ntr} with [T x dx] -- inferred latents evaluated on 1ms grid
% times                     -- times of event to align to 
% times2                    -- other times to plot on figure
% trialLengths [Ntr]        -- length of trial in ms
% ncols                     -- number of columns for subplots
% markerColors              -- colors for markers
% times3                    -- optional third label 
%
% Lea Duncker, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntr = size(mu_fs,3);
dx = size(mu_fs,2);

mu_fs_aligned = alignLatents(mu_fs,times,trialLengths);

maxEventTime = times(times == max(times));
distFromEvent = trialLengths(:) - times(:);
maxDistFromEvent = distFromEvent(distFromEvent== max(distFromEvent));
Tmax = maxDistFromEvent + maxEventTime;
Tmax = Tmax(1);
shiftEvents = times2 - times ;

figure;
ncols = ceil(dx/nrows);
if nargin > 6
    times3 = varargin{1};
    times4 = varargin{2};
end

for jj = 1:dx
    for nn = 1:ntr
        ax(jj) = subplot(nrows,ncols,jj);hold on; plot(mu_fs_aligned(:,jj,nn),'color',[0.3 0.3 0.3]);
        plot(maxEventTime,mu_fs(times(nn),jj,nn),'.','markersize',20,'color',markerColors(1,:))
        plot(maxEventTime + shiftEvents(nn) ,mu_fs(times2(nn),jj,nn),'.','markersize',20,'color',markerColors(2,:))
        xlim([0,Tmax])
    end
    
    
    axis off
    box off
    set(gca,'TickDir','out')
    
end
linkaxes(ax,'xy')

if nargin > 6
    figure;hold on
    for jj = 1:dx
        for nn = 1:ntr
            plot3(mu_fs(:,1,nn),mu_fs(:,2,nn),mu_fs(:,3,nn),'color',[0.3 0.3 0.3]);
            plot3(mu_fs(times(nn),1,nn),mu_fs(times(nn),2,nn),mu_fs(times(nn),3,nn),'.','markersize',20,'color',markerColors(1,:))
            plot3(mu_fs(times2(nn),1,nn),mu_fs(times2(nn),2,nn),mu_fs(times2(nn),3,nn),'.','markersize',20,'color',markerColors(2,:))
            plot3(mu_fs(times3(nn),1,nn),mu_fs(times3(nn),2,nn),mu_fs(times3(nn),3,nn),'.','markersize',20,'color',markerColors(3,:))
            plot3(mu_fs(times4(nn),1,nn),mu_fs(times4(nn),2,nn),mu_fs(times4(nn),3,nn),'.','markersize',20,'color',markerColors(4,:))
            
        end
        box off
        set(gca,'TickDir','out')
    end
    xlabel('$\mathbf{x}_1$','interpreter','latex')
    ylabel('$\mathbf{x}_2$','interpreter','latex')
    zlabel('$\mathbf{x}_3$','interpreter','latex')
    set(gca,'fontsize',18)
    
    figure;hold on
    for jj = 1:dx
        for nn = 1:ntr
            plot(mu_fs(:,1,nn),mu_fs{nn}(:,2,nn),'color',[0.3 0.3 0.3]);
            plot(mu_fs(times(nn),1,nn),mu_fs(times(nn),2,nn),'.','markersize',20,'color',markerColors(1,:))
            plot(mu_fs(times2(nn),1,nn),mu_fs(times2(nn),2,nn),'.','markersize',20,'color',markerColors(2,:))
            plot(mu_fs(times3(nn),1,nn),mu_fs(times3(nn),2,nn),'.','markersize',20,'color',markerColors(3,:))
            plot(mu_fs(times4(nn),1,nn),mu_fs(times4(nn),2,nn),'.','markersize',20,'color',markerColors(4,:))
            
        end
        box off
        set(gca,'TickDir','out')
    end
    xlabel('$\mathbf{x}_1$','interpreter','latex')
    ylabel('$\mathbf{x}_2$','interpreter','latex')
    zlabel('$\mathbf{x}_3$','interpreter','latex')
    set(gca,'fontsize',18)
end