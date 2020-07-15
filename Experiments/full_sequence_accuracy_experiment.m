% This is the full sequence accuracy experiment. The fake is used in its
% entirety/is not interleaved with its real counterpart.
%
% Written by Eleanor Tursman
% Last updated 7/2020
clearvars; close all;

numPCs = 5;
out = zeros(25,4);      % # participants, # scenarios (0 fakes, 1 fake, etc)
out2 = zeros(25,4);
out3 = zeros(25,4);

% Iterates through all 25 participants in the dataset
for i=1:25
    
    data2 = load(['Data/mouth-data-fake2-ID' num2str(i) '.mat']);
    data3 = load(['Data/mouth-data-fake3-ID' num2str(i) '.mat']);
    data4 = load(['Data/mouth-data-fake4-ID' num2str(i) '.mat']);
    
    % Ours
    result = socialVerificationOnlyPCA(data2,data3,data4,1.3);
    fprintf('Ours \t Data: %d\t Result: %d %d %d %d\n',i,result);
    
    % Simple mouth baseline
    result2 = socialVerificationSimpleMouth(data2,data3,data4,1.3);
    fprintf('Simple mouth \t Data: %d\t Result: %d %d %d %d\n',i,result2);
    
    % DWT baseline
    result3 = socialVerNoPCA(data2,data3,data4,1.3);
    fprintf('Wavelet \t Data: %d\t Result: %d %d %d %d\n',i,result3);
    
    out(i,:) = result;
    out2(i,:) = result2;
    out3(i,:) = result3;
    
end

% Prints 0 fakes, 1 fake, 2 fakes, 3 fakes scenarios
fprintf('Accuracy simple mouth: %f %f %f %f\n',mean(out2));
fprintf('Accuracy wavelet: %f %f %f %f\n',mean(out3));
fprintf('Accuracy ours: %f %f %f %f\n',mean(out));

%% Helpers

function [result] = socialVerificationOnlyPCA(data2,data3,data4,thresh)
result = zeros(1,4);
fullLen = min([length(data3.cam1) length(data4.cam1) length(data2.cam1)]);
    
k = 5;
cam1 = cpca(data2.cam1(1:fullLen,:),'k',k);
cam2 = cpca(data2.cam2(1:fullLen,:),'k',k);
cam3 = cpca(data2.cam3(1:fullLen,:),'k',k);
cam4 = cpca(data2.cam4(1:fullLen,:),'k',k);
cam5 = cpca(data2.cam5(1:fullLen,:),'k',k);
cam6 = cpca(data2.cam6(1:fullLen,:),'k',k);
fake2 = cpca(data2.fake(1:fullLen,:),'k',k);
fake3 = cpca(data3.fake(1:fullLen,:),'k',k);
fake4 = cpca(data4.fake(1:fullLen,:),'k',k);


X0 = [cam1.sd  cam2.sd cam3.sd cam4.sd cam5.sd cam6.sd]';
X1 = [cam1.sd  cam2.sd cam3.sd fake4.sd cam5.sd cam6.sd]';
X2 = [cam1.sd  cam2.sd fake3.sd fake4.sd cam5.sd cam6.sd]';
X3 = [cam1.sd  fake2.sd fake3.sd fake4.sd cam5.sd cam6.sd]';

% Test for tracking failures & remove
badInds = find((max(X0,[],1) < 10) == 0);
X0(:,badInds) = [];
X1(:,badInds) = [];
X2(:,badInds) = [];
X3(:,badInds) = [];

[~,tree0] = clusterdata(X0,4);
[~,tree1] = clusterdata(X1,4);
[~,tree2] = clusterdata(X2,4);
[~,tree3] = clusterdata(X3,4);

[numFakes0,~] = detectFakesTree(tree0,thresh);
[numFakes1,o1] = detectFakesTree(tree1,thresh);
[numFakes2,o2] = detectFakesTree(tree2,thresh);
[numFakes3,o3] = detectFakesTree(tree3,thresh);

%%%%%%%%%%%%%%%%%%% 0 fakes case
if (numFakes0 == 0)
    result(1,1) = 1; % TN
end

%%%%%%%%%%%%%%%%%%% 1 fake case
% One fake detected
if (numFakes1 == 1)
    % Check if detected fake is in correct spot
    if (all(o1 == [1;1;1;2;1;1]) || all(o1 == [2;2;2;1;2;2]))
        result(1,2) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 2 fake case
% Two fakes detected
if (numFakes2 == 2)
    % Check if detected fake is in correct spot
    if (all(o2 == [1;1;2;2;1;1]) || all(o2 == [2;2;1;1;2;2]))
        result(1,3) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 3 fake case
% Three fakes detected
if (numFakes3 == 3)
    % Check if detected fake is in correct spot
    if (all(o3 == [1;2;2;2;1;1]) || all(o3 == [2;1;1;1;2;2]))
        result(1,4) = 1; % TP
    end
end

end

function result = socialVerificationSimpleMouth(data2,data3,data4,thresh)
result = zeros(1,4);
fullLen = min([length(data3.cam1) length(data4.cam1) length(data2.cam1)]);
 
d1 = norm(data2.cam1(1:fullLen,15) - data2.cam1(1:fullLen,19));
d2 = norm(data2.cam2(1:fullLen,15) - data2.cam2(1:fullLen,19));
d3 = norm(data2.cam3(1:fullLen,15) - data2.cam3(1:fullLen,19));
d4 = norm(data2.cam4(1:fullLen,15) - data2.cam4(1:fullLen,19));
d5 = norm(data2.cam5(1:fullLen,15) - data2.cam5(1:fullLen,19));
d6 = norm(data2.cam6(1:fullLen,15) - data2.cam6(1:fullLen,19));
d7 = norm(data4.fake(1:fullLen,15) - data4.fake(1:fullLen,19));
d8 = norm(data3.fake(1:fullLen,15) - data3.fake(1:fullLen,19));
d9 = norm(data2.fake(1:fullLen,15) - data2.fake(1:fullLen,19));

X0 = [d1; d2; d3; d4; d5; d6];
X1 = [d1; d2; d3; d7; d5; d6];
X2 = [d1; d2; d8; d7; d5; d6];
X3 = [d1; d9; d8; d7; d5; d6];

[~,tree0] = clusterdata(X0,4);
[~,tree1] = clusterdata(X1,4);
[~,tree2] = clusterdata(X2,4);
[~,tree3] = clusterdata(X3,4);

[numFakes0,~] = detectFakesTree(tree0,thresh);
[numFakes1,o1] = detectFakesTree(tree1,thresh);
[numFakes2,o2] = detectFakesTree(tree2,thresh);
[numFakes3,o3] = detectFakesTree(tree3,thresh);

%%%%%%%%%%%%%%%%%%% 0 fakes case
if (numFakes0 == 0)
    result(1,1) = 1; % TN
end

%%%%%%%%%%%%%%%%%%% 1 fake case
% One fake detected
if (numFakes1 == 1)
    % Check if detected fake is in correct spot
    if (all(o1 == [1;1;1;2;1;1]) || all(o1 == [2;2;2;1;2;2]))
        result(1,2) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 2 fake case
% Two fakes detected
if (numFakes2 == 2)
    % Check if detected fake is in correct spot
    if (all(o2 == [1;1;2;2;1;1]) || all(o2 == [2;2;1;1;2;2]))
        result(1,3) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 3 fake case
% Three fakes detected
if (numFakes3 == 3)
    % Check if detected fake is in correct spot
    if (all(o3 == [1;2;2;2;1;1]) || all(o3 == [2;1;1;1;2;2]))
        result(1,4) = 1; % TP
    end
end

end

function result = socialVerNoPCA(data2,data3,data4,thresh)
result = zeros(1,4);
fullLen = min([length(data3.cam1) length(data4.cam1) length(data2.cam1)]);
 
L = wmaxlev(fullLen,'haar');
[c,~]  = wavedec2(data2.cam1(1:fullLen,:),L,'haar');
[c2,~] = wavedec2(data2.cam2(1:fullLen,:),L,'haar');
[c3,~] = wavedec2(data2.cam3(1:fullLen,:),L,'haar');
[c4,~] = wavedec2(data2.cam4(1:fullLen,:),L,'haar');
[c5,~] = wavedec2(data2.cam5(1:fullLen,:),L,'haar');
[c6,~] = wavedec2(data2.cam6(1:fullLen,:),L,'haar');
[c7,~] = wavedec2(data4.fake(1:fullLen,:),L,'haar');
[c8,~] = wavedec2(data3.fake(1:fullLen,:),L,'haar');
[c9,~] = wavedec2(data2.fake(1:fullLen,:),L,'haar');

X0 = [c; c2; c3; c4; c5; c6];
X1 = [c; c2; c3; c7; c5; c6];
X2 = [c; c2; c8; c7; c5; c6];
X3 = [c; c9; c8; c7; c5; c6];

[~,tree0] = clusterdata(X0,4);
[~,tree1] = clusterdata(X1,4);
[~,tree2] = clusterdata(X2,4);
[~,tree3] = clusterdata(X3,4);

[numFakes0,~]  = detectFakesTree(tree0,thresh);
[numFakes1,o1] = detectFakesTree(tree1,thresh);
[numFakes2,o2] = detectFakesTree(tree2,thresh);
[numFakes3,o3] = detectFakesTree(tree3,thresh);

%%%%%%%%%%%%%%%%%%% 0 fakes case
if (numFakes0 == 0)
    result(1,1) = 1; % TN
end

%%%%%%%%%%%%%%%%%%% 1 fake case
% One fake detected
if (numFakes1 == 1)
    % Check if detected fake is in correct spot
    if (all(o1 == [1;1;1;2;1;1]) || all(o1 == [2;2;2;1;2;2]))
        result(1,2) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 2 fake case
% Two fakes detected
if (numFakes2 == 2)
    % Check if detected fake is in correct spot
    if (all(o2 == [1;1;2;2;1;1]) || all(o2 == [2;2;1;1;2;2]))
        result(1,3) = 1; % TP
    end
end

%%%%%%%%%%%%%%%%%%% 3 fake case
% Three fakes detected
if (numFakes3 == 3)
    % Check if detected fake is in correct spot
    if (all(o3 == [1;2;2;2;1;1]) || all(o3 == [2;1;1;1;2;2]))
        result(1,4) = 1; % TP
    end
end

end