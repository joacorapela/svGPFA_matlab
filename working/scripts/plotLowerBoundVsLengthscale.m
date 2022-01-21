
function plotLowerBoundVsLengthscale(m, nIter, lengthscaleValues, figFilename)
    prs = extract_hyperParams_svGPFA(m);
    maxLengthscale = prs(2);
    for i=1:length(lengthscaleValues)
        prs(2) = lengthscaleValues(i);
        [obj, grad] = hyperMstep_Objective_PointProcess_svGPFA(m, prs, nIter);
        lowerBoundValues(i) = obj;
    end
    plot(lengthscaleValues, -lowerBoundValues, 'o-b');
    y = ylim;
    line([maxLengthscale,maxLengthscale], [y(1),y(2)], 'Color', 'r');
    xlabel('Lengthscale');
    ylabel('LowerBound');
    saveas(gcf, figFilename);

