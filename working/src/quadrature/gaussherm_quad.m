function [x, w] = gaussherm_quad(n,mu, var)
% function to compute quadrature weights to compute Gaussian expectations 
% of arbitrary funtions. The inputs are
% n -- number of nodes
% mu -- mean of Gaussian 
% var -- variance of Gaussian
% outputs are:
% x -- nodes
% w -- weights
%
%
% E_p[f(x)] = sum_i w_i f(x_i) , where p ~ N(mu,var) 

% compute standard quadrature rules
[x,w] = hermquad(n); 

x = sqrt(2 * var) * x + mu;
w = sqrt(pi) * w; 

