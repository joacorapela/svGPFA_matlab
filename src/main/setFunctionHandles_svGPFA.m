function m = setFunctionHandles_svGPFA(m,lik)
% function to set the function handles of the model

% defaults
m.EMfunctions.sampleMinibatch = @sampleMinibatch_svGPFA;
m.EMfunctions.VariationalFreeEnergy = @VariationalFreeEnergy_svGPFA;
m.EMfunctions.HyperMstep_Objective = @hyperMstep_Objective_svGPFA;
m.EMfunctions.InducingPointMstep_Objective = @inducingPointMstep_Objective_svGPFA;
m.EMfunctions.Estep_Update = @Estep_Update_Iterative_svGPFA;
m.EMfunctions.Mstep_Update = @Mstep_Update_Iterative_svGPFA;
m.EMfunctions.extract_hyperParams = @extract_hyperParams_svGPFA;
m.EMfunctions.predict_MultiOutputGP = @predict_MultiOutputGP;

% override defaults and add lik specific stuff
switch lik
    
    case 'Poisson'
        
        m.EMfunctions.likelihood = @expectedLogLik_Poisson;
        m.EMfunctions.gradLik_hprs = @gradLik_hprs_Poisson_svGPFA;
        m.EMfunctions.gradLik_inducingPoints = @gradLik_inducingPoints_Poisson_svGPFA;
        m.EMfunctions.gradLik_variationalPrs = @gradLik_VariationalPrs_Poisson_svGPFA;
        m.EMfunctions.gradLik_modelPrs = @gradLik_ModelPrs_Poisson;
        m.EMfunctions.updateHyperParams = @updateHyperParams_svGPFA;

    case 'PointProcess'
        
        m.EMfunctions.likelihood = @expectedLogLik_PointProcess;
        m.EMfunctions.gradLik_hprs = @gradLik_hprs_PointProcess_svGPFA;
        m.EMfunctions.gradLik_inducingPoints = @gradLik_inducingPoints_PointProcess_svGPFA;
        m.EMfunctions.gradLik_variationalPrs = @gradLik_VariationalPrs_PointProcess_svGPFA;
        m.EMfunctions.gradLik_modelPrs = @gradLik_ModelPrs_PointProcess;

        m.EMfunctions.updateHyperParams = @updateHyperParams_svGPFA;
        m.EMfunctions.HyperMstep_Objective = @hyperMstep_Objective_PointProcess_svGPFA;
        m.EMfunctions.InducingPointMstep_Objective = @inducingPointMstep_Objective_PointProcess_svGPFA;
        m.EMfunctions.Estep_Update = @Estep_Update_Iterative_PointProcess_svGPFA;
        m.EMfunctions.Mstep_Update = @Mstep_Update_Iterative_PointProcess_svGPFA;
        m.EMfunctions.VariationalFreeEnergy = @VariationalFreeEnergy_PointProcess_svGPFA;
        m.EMfunctions.predict_MultiOutputGP_fromSpikes = @predict_MultiOutputGP_fromSpikes;

end

