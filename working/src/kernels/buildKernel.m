function kern = buildKernel(name,hprs,varargin)
% helper function to build convenient input structure for kernels to svGPFA
% framework
% kern = buildKernel(name,hprs) -- single kernel 
% kern = buildKernel(name1,hprs1,name2,hprs2) -- composition kernel
%
% available kernel names: 
%   RBF, Periodic, LocallyPeriodic, Matern32, Matern52, RationalQuadratic, 
%   Linear, RBF+Linear, Periodic+Linear, LocallyPeriodic+Linear 
%   

% see if GPU flag is provided so we can build the psi-stats on a GPU
if nargin < 3
    gpuflag = 0;
else
    gpuflag = varargin{1};
end

switch name
    
    % ======= standard Kernels =======
    
    case 'RBF'
        % general kernel functions
        kern.numhprs = 1;
        kern.hprs = hprs;
        kern.K = @rbfKernel;
        kern.Kdiag = @Kdiag_rbfKernel;
        kern.dKhprs = @dKhprs_rbfKernel;
        kern.dKin = @dKin_rbfKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_rbfKernel;
        kern.Psi1 = @psi1_rbfKernel;
        kern.Psi2 = @psi2_rbfKernel;
        
        % evaluate gradients of psi statistics
        kern.dPsi1hprs = @dpsi1hprs_rbfKernel;
        kern.dPsi2hprs = @dpsi2hprs_rbfKernel;
        kern.dPsi1in = @dpsi1in_rbfKernel;
        kern.dPsi2in = @dpsi2in_rbfKernel;
        kern.dPsi1mu = @dpsi1mu_rbfKernel;
        kern.dPsi2mu = @dpsi2mu_rbfKernel;
        kern.dPsi1sig = @dpsi1sig_rbfKernel;
        kern.dPsi2sig = @dpsi2sig_rbfKernel;
        
    case 'Cosine'
        % general kernel functions
        kern.numhprs = 1;
        kern.hprs = hprs;
        kern.K = @cosineKernel;
        kern.Kdiag = @Kdiag_cosineKernel;
        kern.dKhprs = @dKhprs_cosineKernel;
        kern.dKin = @dKin_cosineKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_cosineKernel;
        kern.Psi1 = @psi1_cosineKernel;
        kern.Psi2 = @psi2_cosineKernel;
        
        % evaluate gradients of psi statistics
        kern.dPsi1hprs = @dpsi1hprs_cosineKernel;
        kern.dPsi2hprs = @dpsi2hprs_cosineKernel;
        kern.dPsi1in = @dpsi1in_cosineKernel;
        kern.dPsi2in = @dpsi2in_cosineKernel;
        kern.dPsi1mu = @dpsi1mu_cosineKernel;
        kern.dPsi2mu = @dpsi2mu_cosineKernel;
        kern.dPsi1sig = @dpsi1sig_cosineKernel;
        kern.dPsi2sig = @dpsi2sig_cosineKernel;
                
    case 'Periodic'
        % general kernel functions
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = @PeriodicKernel;
        kern.Kdiag = @Kdiag_PeriodicKernel;
        kern.dKhprs = @dKhprs_PeriodicKernel;
        kern.dKin = @dKin_PeriodicKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_PeriodicKernel;
        kern.Psi1 = @psi1_PeriodicKernel;
        kern.Psi2 = @psi2_PeriodicKernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_PeriodicKernel;
%         kern.dPsi2hprs = @dpsi2hprs_PeriodicKernel;
%         kern.dPsi1in = @dpsi1in_PeriodicKernel;
%         kern.dPsi2in = @dpsi2in_PeriodicKernel;
%         kern.dPsi1mu = @dpsi1mu_PeriodicKernel;
%         kern.dPsi2mu = @dpsi2mu_PeriodicKernel;
%         kern.dPsi1sig = @dpsi1sig_PeriodicKernel;
%         kern.dPsi2sig = @dpsi2sig_PeriodicKernel;
        
    case 'LocallyPeriodic'
        % general kernel functions
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = @LocallyPeriodicKernel;
        kern.Kdiag = @Kdiag_LocallyPeriodicKernel;
        kern.dKhprs = @dKhprs_LocallyPeriodicKernel;
        kern.dKin = @dKin_LocallyPeriodicKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_LocallyPeriodicKernel;
        kern.Psi1 = @psi1_LocallyPeriodicKernel;
        kern.Psi2 = @psi2_LocallyPeriodicKernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_LocallyPeriodicKernel;
%         kern.dPsi2hprs = @dpsi2hprs_LocallyPeriodicKernel;
%         kern.dPsi1in = @dpsi1in_LocallyPeriodicKernel;
%         kern.dPsi2in = @dpsi2in_LocallyPeriodicKernel;
%         kern.dPsi1mu = @dpsi1mu_LocallyPeriodicKernel;
%         kern.dPsi2mu = @dpsi2mu_LocallyPeriodicKernel;
%         kern.dPsi1sig = @dpsi1sig_LocallyPeriodicKernel;
%         kern.dPsi2sig = @dpsi2sig_LocallyPeriodicKernel;
%         
    case 'Matern32'
        % general kernel functions
        kern.numhprs = 1;
        kern.hprs = hprs;
        kern.K = @matern32Kernel;
        kern.Kdiag = @Kdiag_matern32Kernel;
        kern.dKhprs = @dKhprs_matern32Kernel;
        kern.dKin = @dKin_matern32Kernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_matern32Kernel;
        kern.Psi1 = @psi1_matern32Kernel;
        kern.Psi2 = @dpsi2_matern32Kernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_matern32Kernel;
%         kern.dPsi2hprs = @dpsi2hprs_matern32Kernel;
%         kern.dPsi1in = @dpsi1in_matern32Kernel;
%         kern.dPsi2in = @dpsi2in_matern32Kernel;
%         kern.dPsi1mu = @dpsi1mu_matern32Kernel;
%         kern.dPsi2mu = @dpsi2mu_matern32Kernel;
%         kern.dPsi1sig = @dpsi1sig_matern32Kernel;
%         kern.dPsi2sig = @dpsi2sig_matern32Kernel;
%         
        
    case 'Matern52'
        % general kernel functions
        kern.numhprs = 1;
        kern.hprs = hprs;
        kern.K = @matern52Kernel;
        kern.Kdiag = @Kdiag_matern52Kernel;
        kern.dKhprs = @dKhprs_matern52Kernel;
        kern.dKin = @dKin_matern52Kernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_matern52Kernel;
        kern.Psi1 = @psi1_matern52Kernel;
        kern.Psi2 = @dpsi2_matern52Kernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_matern52Kernel;
%         kern.dPsi2hprs = @dpsi2hprs_matern52Kernel;
%         kern.dPsi1in = @dpsi1in_matern52Kernel;
%         kern.dPsi2in = @dpsi2in_matern52Kernel;
%         kern.dPsi1mu = @dpsi1mu_matern52Kernel;
%         kern.dPsi2mu = @dpsi2mu_matern52Kernel;
%         kern.dPsi1sig = @dpsi1sig_matern52Kernel;
%         kern.dPsi2sig = @dpsi2sig_matern52Kernel;
        
        
    case 'RationalQuadratic'
        % general kernel functions
        kern.numhprs = 1;
        kern.hprs = hprs;
        kern.K = @RationalQuadraticKernel;
        kern.Kdiag = @Kdiag_RationalQuadraticKernel;
        kern.dKhprs = @dKhprs_RationalQuadraticKernel;
        kern.dKin = @dKin_RationalQuadraticKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_RationalQuadraticKernel;
        kern.Psi1 = @psi1_RationalQuadraticKernel;
        kern.Psi2 = @dpsi2_RationalQuadraticKernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_RationalQuadraticKernel;
%         kern.dPsi2hprs = @dpsi2hprs_RationalQuadraticKernel;
%         kern.dPsi1in = @dpsi1in_RationalQuadraticKernel;
%         kern.dPsi2in = @dpsi2in_RationalQuadraticKernel;
%         kern.dPsi1mu = @dpsi1mu_RationalQuadraticKernel;
%         kern.dPsi2mu = @dpsi2mu_RationalQuadraticKernel;
%         kern.dPsi1sig = @dpsi1sig_RationalQuadraticKernel;
%         kern.dPsi2sig = @dpsi2sig_RationalQuadraticKernel;
        
        
    case 'Linear'
        % general kernel functions
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = @LinearKernel;
        kern.Kdiag = @Kdiag_LinearKernel;
        kern.dKhprs = @dKhprs_LinearKernel;
        kern.dKin = @dKin_LinearKernel;
        
        % evaluate psi statistics
        kern.Psi0 = @psi0_LinearKernel;
        kern.Psi1 = @psi1_LinearKernel;
        kern.Psi2 = @dpsi2_LinearKernel;
        
        % evaluate gradients of psi statistics
%         kern.dPsi1hprs = @dpsi1hprs_LinearKernel;
%         kern.dPsi2hprs = @dpsi2hprs_LinearKernel;
%         kern.dPsi1in = @dpsi1in_LinearKernel;
%         kern.dPsi2in = @dpsi2in_LinearKernel;
%         kern.dPsi1mu = @dpsi1mu_LinearKernel;
%         kern.dPsi2mu = @dpsi2mu_LinearKernel;
%         kern.dPsi1sig = @dpsi1sig_LinearKernel;
%         kern.dPsi2sig = @dpsi2sig_LinearKernel;
        
    % ======= composition Kernels ======= 
    
    case 'RBF+Linear'
        kern.numhprs = 5;
        kern.hprs = hprs;
        kern.K = @rbf_and_linearKernel;
        kern.Kdiag = @Kdiag_rbf_and_linearKernel;
        kern.dKhprs = @dKhprs_rbf_and_linearKernel;
        kern.dKin = @dKin_rbf_and_linearKernel;
        
    case 'Periodic+Linear'
        kern.numhprs = 6;
        kern.hprs = hprs;
        kern.K = @Periodic_and_linearKernel;
        kern.Kdiag = @Kdiag_Periodic_and_linearKernel;
        kern.dKhprs = @dKhprs_Periodic_and_linearKernel;
        kern.dKin = @dKin_Periodic_and_linearKernel;
        
    case 'LocallyPeriodic+Linear'
        kern.numhprs = 7;
        kern.hprs = hprs;
        kern.K = @LocallyPeriodic_and_linearKernel;
        kern.Kdiag = @Kdiag_LocallyPeriodic_and_linearKernel;
        kern.dKhprs = @dKhprs_LocallyPeriodic_and_linearKernel;
        kern.dKin = @dKin_LocallyPeriodic_and_linearKernel;             
end
