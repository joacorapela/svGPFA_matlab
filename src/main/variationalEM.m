function m = variationalEM(m);
% m = variationalEM(m);
%
% this function fits a sparse variational GPFA model to multivariate
% observations using variational EM
%
% input:
% ======
% m         -- model structure created using InitialiseModel_svGPFA (or
%              (other model initialiser)
%
% output:
% ======
% m         -- model structure with optimised parameters
%
% See also: InitialiseModel_svGPFA, InitialiseModel_grouped_svGPFA,
%           InitialiseModel_warped_grouped_svGPFA
%
%
% Duncker, 2018
%
%

saveCItestData = getGlobal_saveCItestData();
if saveCItestData
    q_mu0 = m.q_mu;
    q_sqrt0 = m.q_sqrt;
    q_diag0 = m.q_diag;
    C0 = m.prs.C;
    b0 = m.prs.b;
    index = m.index;
    ttQuad = m.ttQuad;
    wwQuad = m.wwQuad;
    xxHerm = m.xxHerm;
    wwHerm = m.wwHerm;
    Z0 = m.Z;
    YNonStacked = m.YNonStacked;
    epsilon = m.epsilon;
    hprs0 = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';
    kernelNames = {};
    for k=1:length(m.kerns)
        kernelNames{k} = func2str(m.kerns{k}.K);
    end
end

if m.savePartial 
    savePartialFilename = sprintf(m.savePartialFilenamePattern, 'initial');
    save(savePartialFilename, 'm');
end

t_start = tic; % record starting time
abstol = 1e-05; % convergence tolerance
m.FreeEnergy = [m.EMfunctions.VariationalFreeEnergy(m);];

% print output
if m.opts.verbose
    fprintf('%3s\t%10s\t%10s\t%10s\n','iter','objective','increase','iterTime')
end

% ========= run variational inference =========
t_start_iter = tic;
for i = 1:m.opts.maxiter.EM

    % ========= resample minibatch ============

    m = m.EMfunctions.sampleMinibatch(m);

    % ========= E-step: update variational parameters =========

    m = Estep(m);

    if m.savePartial
        savePartialFilename = sprintf(m.savePartialFilenamePattern, sprintf('eStep%03d', i));
        save(savePartialFilename, 'm');
    end

    % ========= compute new value of free energy ==================

    m.FreeEnergy(i,1) = m.EMfunctions.VariationalFreeEnergy(m);
    m.iterationTime(i,1) = toc(t_start_iter);

    if i > 1
        FEdiff = m.FreeEnergy(i,1) - m.FreeEnergy(i-1,1);
    else
        FEdiff = NaN;
    end

    if m.opts.verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n',i,m.FreeEnergy(i),FEdiff,m.iterationTime(i));
    end

    % ========= check convergence in free energy =========
    if i > 2 && abs(m.FreeEnergy(i,1) - m.FreeEnergy(i-1,1)) < abstol
        break;
    end
    t_start_iter = tic;

    % ========= M-step: optimise wrt model parameters =========

    if i > 1 % skip first M-step to avoid early convergence to bad optima
        m = Mstep(m);
        if m.savePartial
            savePartialFilename = sprintf(m.savePartialFilenamePattern, sprintf('mStepEmbedding%03d', i));
            save(savePartialFilename, 'm');
        end
    end

    % ========= hyper-M step: optimise wrt hyperparameters =========

    m = hyperMstep(m);
    if m.savePartial
        savePartialFilename = sprintf(m.savePartialFilenamePattern, sprintf('mStepKernels%03d', i));
        save(savePartialFilename, 'm');
    end

    % ========= inducing point hyper-M step: optimise wrt inducing point locations =========

    m = inducingPointMstep(m);
    if m.savePartial
        savePartialFilename = sprintf(m.savePartialFilenamePattern, sprintf('mStepIndPoints%03d', i));
        save(savePartialFilename, 'm');
    end

    m.elapsedTime(i,1) = toc(t_start);
end

% save and report elapsed time
m.RunTime = toc(t_start);
if m.opts.verbose
    fprintf('Elapsed time is %1.5f seconds\n',m.RunTime);
end

if saveCItestData
    filename = 'results/ciTestData/variationalEM.mat';
    lowerBound = m.FreeEnergy(end,1);
    save(filename, 'epsilon', 'q_mu0', 'q_sqrt0', 'q_diag0', 'C0', 'b0', 'index', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z0', 'YNonStacked', 'hprs0', 'kernelNames', 'lowerBound');
end
