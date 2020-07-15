% Helper which, given a binary tree and a threshold value, chooses where to
% cut a tree to separate real/fake leaves. numFakes will be the number of
% fakes detected, and c will be a vector of numCams integers, which are 
% partitioned into two sets. We assume the larger partition is real.
%
% Written by Eleanor Tursman
% Last updated 7/2020

function [numFakes,c] = detectFakesTree(tree,thresh)

ratio = tree(end,end) / tree(end-1,end);
if (ratio > thresh)
    % we have at least one fake
    c = cluster(tree,2);
    
    partition1 = length(find(c == 1));
    partition2 = length(find(c == 2));
    
    if(partition1 > partition2)
        % partition 1 is real
        numFakes = partition2;
    else
        numFakes = partition1;
    end
    
else
    % no fakes detected
    numFakes = 0;
    c = 0;
end

end