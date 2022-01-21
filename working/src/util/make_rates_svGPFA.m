function rates = make_rates_svGPFA(C,b,fs,nonlin)
% returns cell array with function handles to rates
[D,~] = size(C);
[~,ntr] = size(fs); % fs is function handle with latent for each trial
rates = cell(D,ntr);

for m = 1:ntr
    for d = 1:D
        rates{d,m} = @(tt) single_rate(tt,d,C,b,fs(:,m),nonlin);
    end
end

end

function rate_n  = single_rate(tt,n,C,b,fs,nonlin);
% returns rate for n-th neuron
[~,K] = size(C);
Ff = zeros(K,length(tt));
if ~isempty(Ff)
    for i = 1:K
        Ff(i,:)  = fs{i}(tt);
    end
end
rate_n = nonlin(C(n,:)*Ff + b(n));
end