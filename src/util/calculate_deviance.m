function dev = calculate_deviance(Y,lambda,dt);
% function to calculate the deviance of a poisson model with inhomogenous
% rate parameter lambda and bin size dt
% lambda: N x T x R
% Y: N x T x R

dev = -2* ( -dt * sum(lambda(:)) + sum(Y(:)) + Y(:)'*log(lambda(:).*dt) -  Y(Y~=0)'*log(Y(Y~=0)));