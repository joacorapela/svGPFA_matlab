function [fx,dfx] = relu(x,a);

fx = max(x,a);

dfx = double(fx > a);