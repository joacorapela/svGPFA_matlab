function [fx,dfx] = sigmoid(x,a);

fx = exp(a*x)./(1 + exp(a*x));

dfx = a*exp(a*x) .* ((1 + exp(a*x)) - a*exp(a*x))./(1 + exp(a*x)).^2;