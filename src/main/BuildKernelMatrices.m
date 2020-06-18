function KMats = BuildKernelMatrices(m,tt,Z,hprs,flag);
% build the kernel matrices needed for Gaussian svGPFA
% flag indicates whether gradient matrices should also be returned
% flag = 0 - no gradients
% flag = 1 - gradients wrt hyperparameters
% flag = 2 - gradeints wrt inducing points
%

% pre-allocate cell array for saving kernel matrices
Kzz = cell(m.dx,1);
Kzzi = cell(m.dx,1);
Ktz = cell(m.dx,1);
Ktt = zeros(size(tt,1),m.dx,size(tt,3));

% pre-allocate space for gradients
if flag == 1
    dKtzhprs = cell(m.dx,1);
    dKtt = cell(m.dx,1);
    dKzzhprs = cell(m.dx,1);
elseif flag ==2
    dKzzin = cell(m.dx,1);
    dKtzin = cell(m.dx,1);
end

for k = 1:m.dx
    if flag == 1
        [Ktt(:,k,:),dKtt{k}] = m.kerns{k}.Kdiag(hprs{k},tt);
    else
        Ktt(:,k,:) = m.kerns{k}.Kdiag(hprs{k},tt);
    end
    
    Kzz{k} = m.kerns{k}.K(hprs{k},Z{k}) + m.epsilon*eye(m.numZ(k),m.numZ(k));
    Kzzi{k} = pinv3D(Kzz{k});
    Ktz{k} = m.kerns{k}.K(hprs{k},tt,Z{k});
    if flag == 1
        dKzzhprs{k} = m.kerns{k}.dKhprs(hprs{k},Z{k});
        dKtzhprs{k} = m.kerns{k}.dKhprs(hprs{k},tt,Z{k});
    elseif flag ==2
        dKtzin{k} = m.kerns{k}.dKin(m.kerns{k}.hprs,tt,Z{k});
        [~,dKzzin{k}] = m.kerns{k}.dKin(m.kerns{k}.hprs,Z{k});
    end
end

KMats.Kzz = Kzz;
KMats.Kzzi = Kzzi;
KMats.Ktz = Ktz;
KMats.Ktt = Ktt;

if flag == 1
    KMats.dKzzhprs = dKzzhprs;
    KMats.dKtzhprs = dKtzhprs;
    KMats.dKtt = dKtt;
elseif flag == 2
    KMats.dKtzin = dKtzin;
    KMats.dKzzin = dKzzin;
end

kernelNames = {};
for k=1:length(m.kerns)
    kernelNames{k} = func2str(m.kerns{k}.K);
end

filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/BuildKernelMatrices.mat';
epsilon = m.epsilon;
save(filename, 'epsilon', 'tt', 'Z', 'kernelNames', 'hprs', 'Kzz', 'Kzzi', 'Ktz', 'Ktt');

keyboard
