function KMats = BuildKernelMatrices_fromSpikes(m,Z,hprs,flag,ntr);
% build the kernel matrices needed for PointProcess svGPFA
% flag indicates whether gradient matrices should also be returned
% flag = 0 - no gradients
% flag = 1 - gradients wrt hyperparameters
% flag = 2 - gradeints wrt inducing points
%
% This function builds cell arrays of kernel matrices of different lengths
% for each trial and doesn't build Kzz or Kzzi
%
if nargin < 5 % evalutate over all trials
    trEval = 1:m.ntr;
    timeEval = 1:m.ntr;
elseif length(ntr) == 1
    trEval = 1;
    timeEval = ntr;
else
    trEval = ntr;
    timeEval(1:length(trEval)) = ntr; 
end

R = length(trEval); % number of trials to evaluate

% pre-allocate cell array for saving kernel matrices
Ktz = cell(m.dx,R);
Ktt = cell(m.dx,R);

% pre-allocate space for gradients
if flag == 1
    dKtzhprs = cell(m.dx,R);
    dKtt = cell(m.dx,R);
elseif flag ==2
    dKtzin = cell(m.dx,R);
end

for nn = 1:R
    for k = 1:m.dx
        if flag == 1
            [Ktt{k,nn},dKtt{k,nn}] = m.kerns{k}.Kdiag(hprs{k},m.Y{timeEval(nn)});
        else
            Ktt{k,nn} = m.kerns{k}.Kdiag(hprs{k},m.Y{timeEval(nn)});
        end
        Ktz{k,nn} = m.kerns{k}.K(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,nn));
        if flag == 1
            dKtzhprs{k,nn} =  m.kerns{k}.dKhprs(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,nn));
        elseif flag ==2
            dKtzin{k,nn} = m.kerns{k}.dKin(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,nn));
        end
    end
end

KMats.Ktz = Ktz;
KMats.Ktt = Ktt;

if flag == 1
    KMats.dKtzhprs = dKtzhprs;
    KMats.dKtt = dKtt;
elseif flag == 2
    KMats.dKtzin = dKtzin;
end

kernelNames = {};
for k=1:length(m.kerns)
    kernelNames{k} = func2str(m.kerns{k}.K);
end

% Y = m.Y;
% 
% filename = '~/dev/research/gatsby/svGPFA/code/test/data/BuildKernelMatrices_fromSpikes.mat';
% save(filename, 'Y', 'Z', 'kernelNames', 'hprs', 'Ktz', 'Ktt');
% 
% keyboard
% 
