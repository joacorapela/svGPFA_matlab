function [fx,dfx] = softrec(x,a)

fx = 1/a * log( 1 + exp(a*x));

dfx = sigmoid(x,a);