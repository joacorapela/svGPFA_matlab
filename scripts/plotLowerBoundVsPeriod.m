
function plotLowerBoundVsPeriod(m, nIter, periodValues, figFilename)
    prs = extract_hyperParams_svGPFA(m);
    maxPeriod = prs(2);
    for i=1:length(periodValues)
        prs(2) = periodValues(i);
        [obj, grad] = hyperMstep_Objective_PointProcess_svGPFA(m, prs, nIter);
        lowerBoundValues(i) = obj;
    end
    plot(periodValues, -lowerBoundValues);
    y = ylim;
    line([maxPeriod,maxPeriod], [y(1),y(2)], 'Color', 'r');
    xlabel('Period');
    ylabel('LowerBound');
    saveas(gcf, figFilename);

