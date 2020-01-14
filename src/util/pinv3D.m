function Kinv = pinv3D(K);

Kinv = zeros(size(K));
for ii = 1:size(K,3);
    Kinv(:,:,ii) = pinv(K(:,:,ii));
end