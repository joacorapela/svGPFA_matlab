clear all
% load data
load('/Users/leaduncker/Documents/Gatsby/projects/cosyne/neural_data_example.mat')
cd('/Users/leaduncker/Documents/git/svGPFA/')
set_paths
cd('/Users/leaduncker/Documents/Gatsby/projects/svGPFA/reaching_task/scripts/')
%%
[Ntrials,Ntargets] = size(Rb);
sps = cellfun(@(x){x.spikeTimes}',({(Rb.unit)}),'uni',0);
sps = [sps{:}];
sps = reshape(sps,[size(sps,1), Ntrials,Ntargets]);

% convert to seconds
sps = cellfun(@(x) x./1000,sps,'uni',0);
%% extract trial variables
nTarget = 9;

trialLengths = reshape([Rb.trialLength]',[Ntrials,Ntargets])/1000;
timeGoCue = reshape([Rb.timeGoCuePHOTO]',[Ntrials,Ntargets])/1000;
timeMoveOnset = reshape([Rb.timeMoveOnset]',[Ntrials,Ntargets])/1000;
timeMoveEnd = reshape([Rb.timeMoveEnd]',[Ntrials,Ntargets])/1000;
timeTargetOnset = reshape([Rb.timeTouchHeldPHOTO]',[Ntrials,Ntargets])/1000;

trialLengths = trialLengths(:,nTarget);
timeGoCue = timeGoCue(:,nTarget);
timeMoveOnset = timeMoveOnset(:,nTarget);
timeMoveEnd = timeMoveEnd(:,nTarget);
timeTargetOnset = timeTargetOnset(:,nTarget);

% time line for evaluating predictions
ttEval = linspace(0,max(trialLengths),2000);

load('~/Documents/Gatsby/projects/svGPFA/reaching_task/pp_svGPFA/svGPFA_pointprocess_TargDir09_numTrials_45_numLatents_6_numInducing_20_numQuad_300_fixInducing_1_30-Sep-2017.mat')
%% get events in terms of indices on ttEval axis
[~,EventTimes] = time2ind(ttEval,[trialLengths,timeGoCue,timeMoveOnset,timeMoveEnd,timeTargetOnset]);
ttLen = EventTimes(:,1);
ttGC = EventTimes(:,2);
ttMO = EventTimes(:,3);
ttME = EventTimes(:,4);
ttTO = EventTimes(:,5);

%% 
addpath('~/Documents/MATLAB/plotting/')
dy = size(m.prs.C,1);
dx = size(m.prs.C,2);

cols = cbrewer('qual', 'Set1', 5);
nn = 10;
figure; imagesc(ttEval(ttTO(nn)-100:ttMO(nn)+50),1:dy,exp(pred.multiOutputGP.mean(ttTO(nn)-100:ttMO(nn)+50,:,nn)'))
colormap parula
xlabel('time in ms')
ylabel('neuron number')
hold on
plot(repmat(ttEval(ttTO(nn)),[1 dy]),1:dy,'color',cols(1,:),'linewidth',2)
plot(repmat(ttEval(ttGC(nn)),[1 dy]),1:dy,'color',cols(5,:),'linewidth',2)
plot(repmat(ttEval(ttMO(nn)),[1 dy]),1:dy,'color',cols(3,:),'linewidth',2)
plot(repmat(ttEval(ttME(nn)),[1 dy]),1:dy,'color',cols(4,:),'linewidth',2)
set(gca,'fontsize',18,'TickDir','out','Ytick',[1 105])
box off


%%
cd('~/Documents/Gatsby/projects/svGPFA/reaching_task/scripts/')
%%
[mu_fs_orth,Corth] = orthonomalisedLatents(m.prs.C,pred.latents.mean,ntr);
mu_fs = pred.latents.mean;


%% plot latents for trials
% figure; 
ntr = 45;
for ii = 1:ntr;
    for jj = 1
        hold on;
        ylabel('$x_1(t)$','fontsize',18,'interpreter','latex')
        hold on; 
%         plot(pred.latents.mean(:,jj,ii));
        plot(ttEval,mu_fs_orth(:,jj,ii))
        hold on; plot(m.Z{jj}(:,:,ii),min(pred.latents.mean(:,jj,ii))*ones(size(m.Z{jj}(:,:,ii))),'x')
        fprintf('trial %d \n',ii);
        pause(.1)
    end 
end

%% GC aligned
% tridx = [1:17,19:24,26:ntr];
tridx = 1:ntr;
cols = cbrewer('qual', 'Set1', 5);
markerColors = cols([5,3],:);
plotAlignedLatents(mu_fs_orth,ttGC(tridx),ttMO(tridx),ttLen(tridx),2,markerColors);
% plotAlignedLatents(mu_fs,ttGC(tridx),ttMO(tridx),ttLen(tridx),2,markerColors);

%% MO aligned
cols = cbrewer('qual', 'Set1', 5);
markerColors = cols([3,5],:);
plotAlignedLatents(mu_fs,ttMO(tridx),ttGC(tridx),ttLen(tridx),2,markerColors);
%%
[interTrialMeanGC,interTrialStdGC,tt_GC,mEndGC] = GetLatentScatter(mu_fs_orth,ttGC(tridx),ttLen(tridx),2);
[interTrialMeanMO,interTrialStdMO,tt_MO,mEndMO] = GetLatentScatter(mu_fs_orth,ttMO(tridx),ttLen(tridx),2);
%%

cols = cbrewer('div','RdBu',10);

meanSTD_ct = [ nanmean(vec(interTrialStdGC(tt_GC:end,1:end))) nanmean(vec(interTrialStdMO(tt_GC:end,1:end)))];
meanSTDerr_ct = [ nanstd(vec(interTrialStdGC(tt_GC:end,1:end))) nanstd(vec(interTrialStdMO(tt_GC:end,1:end)))];

figure; h = bar(meanSTD_ct);
set(gca,'XTickLabel',{'GC aligned','MO aligned'})
set(h(1),'FaceColor',cols(9,:));
ylabel('std. deviation across trials')

%%
bar([(nanmean(interTrialStdGC(tt_GC:end,1:end),1))' (nanmean(interTrialStdMO(tt_GC:end,1:end),1))'])
%%
figure;
subplot(311); plot(mean(interTrialStdGC,2)); hold on; plot(mean(interTrialStdMO,2))
plot(tt_GC,mean(interTrialStdGC(tt_GC,:),2),'.','markersize',20,'color',[0.8 0.5 0.0])
plot(tt_MO,mean(interTrialStdMO(tt_MO,:),2),'.','markersize',20,'color',[0.4 0.1 0.6])
title('av. per-time-point std. deviation')
ax1 = gca;

subplot(312); plot((1:size(interTrialStdGC1,1))*20,mean(interTrialStdGC1,2)); hold on;
plot((1:size(interTrialStdGC1,1))*20,mean(interTrialStdMO1,2))
plot(tt_GC1*20,mean(interTrialStdGC1(tt_GC1,:),2),'.','markersize',20,'color',[0.8 0.5 0.0])
plot(tt_MO1*20,mean(interTrialStdMO1(tt_MO1,:),2),'.','markersize',20,'color',[0.4 0.1 0.6])
ax2 = gca;

subplot(313); plot((1:size(interTrialStdGC2,1))*50,mean(interTrialStdGC2,2)); hold on; 
plot((1:size(interTrialStdGC2,1))*50,mean(interTrialStdMO2,2))
plot(tt_GC2*50,mean(interTrialStdGC2(tt_GC2,:),2),'.','markersize',20,'color',[0.8 0.5 0.0])
plot(tt_MO2*50,mean(interTrialStdMO2(tt_MO2,:),2),'.','markersize',20,'color',[0.4 0.1 0.6])
ax3 = gca;

linkaxes([ax1 ax2 ax3],'xy')
%%
h3 = barwitherr([meanSTDerr_GPFA50' meanSTDerr_GPFA20' meanSTDerr_ct'],[meanSTD_GPFA50' meanSTD_GPFA20' meanSTD_ct']);
set(h3(1,1),'FaceColor',cols(9,:));
set(h3(1,2),'FaceColor',cols(8,:));
set(h3(1,end),'FaceColor',cols(1,:));
set(gca,'XTickLabel',{'GC aligned','MO aligned'},'Fontsize',18)
legend('GPFA 50ms','GPFA 20ms','sv-ppGPFA')
ylabel('av per-time-point std across latents')
%%
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'latent_alignment_GPFA','-dpdf')

%%
figure; plot(interTrialStdGC(:,:),'r'); hold on; plot(interTrialStdMO(:,:),'b')

%%
figure;
bar([ nanmean(vec(interTrialStdGC(tt_GC:end,1:end))) nanmean(vec(interTrialStdMO(tt_GC:end,1:end)));...
    nanmean(vec(interTrialStdGC(tt_GC:end,1:4))) nanmean(vec(interTrialStdMO(tt_GC:end,1:4)));...
    nanmean(vec(interTrialStdGC(tt_GC:end,1:5))) nanmean(vec(interTrialStdMO(tt_GC:end,1:5)))])
