function [Ystack,trLenN,mask,tsamp,dt] = arrangeData_Poisson(m,Y,trLen);

ntr = m.ntr;
tmax = max(trLen); % maximum length of trial in time
[dy,trLenN] = cellfun(@size,Y); % trial lengths in units of samples
trLenMax = max(trLenN); % maximum trial length in units of samples

% pre-allocate space
Ystack = zeros(dy(1),trLenMax,ntr);

% zero padded matrix
for nn = 1:ntr
    Ystack(:,1:trLenN(nn),nn) = Y{nn};
end

dt = (tmax/trLenMax); % check this
tsamp = linspace(dt,tmax,trLenMax);

% start modified by Joaco 09/03/19
% mask =  bsxfun(@(x,y) x < y,repmat(trLenN,size(Ystack,2),1), (1:size(Ystack,2))');
mask =  bsxfun(@(x,y) x < y,repmat(trLenN',size(Ystack,2),1), (1:size(Ystack,2))');
% end modified by Joaco 09/03/19
