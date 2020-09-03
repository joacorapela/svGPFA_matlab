function sps = simulate_spikes_from_rates(rates,T,ngrid);
% simulate_spikes_ppGPFA -- function to simulate spike trains from a point
% process GPFA model
% input:
% -------
% rates {n,m}           -- cell array containing function handles for rates
%
% T     (1xM) vector    -- end point of interval [0,T] in which to generate
%                          spike train for each trial
% nonlin      handle    -- function handle for non-linearity
% ngrid (1xM) scalar    -- number of points to use for evaluating intensity
%
% output: N number of neurons, M number of trials
% -------
% sps  {NxM} cell       -- spike train for each neuron, M trials
% rate {NxM} cell       -- array of function handles with rate functions
%                          for each neuron, M trials
%
% Duncker, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[D,ntr] = size(rates);
sps = cell(D,ntr);

for m = 1:ntr
    for i = 1:D
        sps{i,m} = sample_inhomogenous(rates{i,m},T(m),ngrid)';
    end
end

end


function x = sample_inhomogenous(rr,T,ngrid)
% generate a nonhomogeneousl poisson process on [0,T] with intensity function intens
x = linspace(0,T,ngrid);
l = rr(x);
lam0 = max(l); % generate homogeneouos poisson process
u = rand(1,ceil(1.5*T*lam0));
x = cumsum(-(1/lam0)*log(u)); % points of homogeneous pp

x = x(x<T); n=length(x); % select those points less than T

l = rr(x); % evaluates intensity function

x = x(rand(1,n)<l/lam0); % filter out some points
end

