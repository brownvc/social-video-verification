% Runs PCA & calculates the Mahalanobis distance for three separate
% cameras, concats them, and removes spikes in the output caused by
% landmark tracking failures.
%
% camA and camB are the 2D landmark data matrices for real cameras 
% fake is the 2D landmark data matrix for a fake camera
% out is a matrix of the Mahalanobis distances of these cameras
%
% Written by Eleanor Tursman
% Last updated 7/2020

function out = angularHelper(camA,camB,fake)

camAOut = cpca(camA,'k',5);
camBOut = cpca(camB,'k',5);
camFake = cpca(fake,'k',5);

out = [camAOut.sd camBOut.sd camFake.sd]';

% Test for tracking failures & remove
badInds = find((max(out,[],1) < 10) == 0);
out(:,badInds) = [];

end