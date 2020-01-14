function y = rescale_vector(x,a,b);
% function to rescale a vector into the range [a,b]

y = (b-a) * (x - min(x)) ./(max(x) - min(x)) + a;