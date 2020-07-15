% This experiment saves TP, FP, FN, and TN rates for a chosen detection
% method vs 0, 1, 2, and 3 lipgan fakes present in a set of videos. 
%
% Written by Eleanor Tursman
% Last updated 7/2020

clearvars; close all;
fprintf('Stress Test: Interleaved Baselines \n');

method = 'onlyPCA'; %options: 'onlyPCA', 'simpleMouth', 'noPCA'

% Iterate over people
people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
for p=1:length(people)
    
    person = people{p};
    fprintf('Current person: %s\n', person);
    
    % eg: fake3 indicates that the fake in the struct is the lipgan output  
    % of camera 3
    data3 = load(['Data/mouth-data-fake3-ID' person '.mat']);
    data2 = load(['Data/mouth-data-fake2-ID' person '.mat']);
    data4 = load(['Data/mouth-data-fake4-ID' person '.mat']);
    
    fullLen = min([length(data3.cam1) length(data4.cam1) length(data2.cam1)]);
    
    cam1 = data3.cam1(1:fullLen,:);
    cam2 = data3.cam2(1:fullLen,:);
    cam3 = data3.cam3(1:fullLen,:);
    cam4 = data3.cam4(1:fullLen,:);
    cam5 = data3.cam5(1:fullLen,:);
    cam6 = data3.cam6(1:fullLen,:);
    
    % Split into thirds: [fake | real | fake]
    intervalWin = floor(fullLen / 3);
    fake3 = [data3.fake(1:intervalWin,:); cam3((intervalWin + 1):(2*intervalWin),:); data3.fake((2*intervalWin + 1):fullLen,:)];
    fake4 = [data4.fake(1:intervalWin,:); cam4((intervalWin + 1):(2*intervalWin),:); data4.fake((2*intervalWin + 1):fullLen,:)];
    fake2 = [data2.fake(1:intervalWin,:); cam2((intervalWin + 1):(2*intervalWin),:); data2.fake((2*intervalWin + 1):fullLen,:)];
    
    % calculate L2 between real/fake mouth landmarks for all frames
    baseline = vecnorm(cam4 - fake4, 2, 2);
    
    % Iterate over thresholds too
    threshes = [1.1 1.3 1.5 1.7 1.9 2.1];
    
    mkdir(['Output/ID' person])
    mkdir(['Output/ID' person '/thresh_1/'])
    mkdir(['Output/ID' person '/thresh_2/'])
    mkdir(['Output/ID' person '/thresh_3/'])
    mkdir(['Output/ID' person '/thresh_4/'])
    mkdir(['Output/ID' person '/thresh_5/'])
    mkdir(['Output/ID' person '/thresh_6/'])
    
    parfor t=1:length(threshes) %parfor on/off for this line
        thresh = threshes(t);
        
        for i=[50 150 250 350]
            fprintf('Window size: %d\n',i);
            
            numWin = fullLen - i;
            
            % format: row 1- TP, row 2- TN, row 3- FP, row 4- FN
            acc0 = zeros(4,numWin);
            acc1 = zeros(4,numWin);
            acc2 = zeros(4,numWin);
            acc3 = zeros(4,numWin);
            base = zeros(4,numWin);
            
            for startF=1:fullLen

                endF = startF + i;
                curRange = startF:endF; 
                
                % Check if we go over the max number of frames
                if (endF > fullLen)
                    continue;
                end
                
                % Use one of three methods to detect fakes
                if (strcmp(method,'onlyPCA'))
                    % hierarchical clustering directly on mahalanobis
                    % distances
                    
                    k = 5;
                    cam1Out = cpca(cam1(curRange,:),'k',k);
                    cam2Out = cpca(cam2(curRange,:),'k',k);
                    cam3Out = cpca(cam3(curRange,:),'k',k);
                    cam4Out = cpca(cam4(curRange,:),'k',k);
                    cam5Out = cpca(cam5(curRange,:),'k',k);
                    cam6Out = cpca(cam6(curRange,:),'k',k);
                    camFake1 = cpca(fake2(curRange,:),'k',k);
                    camFake2 = cpca(fake3(curRange,:),'k',k);
                    camFake3 = cpca(fake4(curRange,:),'k',k);
                    
                    % X0 is no fakes, X1 one fake, etc.
                    X0 = [cam1Out.sd  cam2Out.sd cam3Out.sd cam4Out.sd cam5Out.sd cam6Out.sd]';
                    X1 = [cam1Out.sd  cam2Out.sd cam3Out.sd camFake3.sd cam5Out.sd cam6Out.sd]';
                    X2 = [cam1Out.sd  cam2Out.sd camFake2.sd camFake3.sd cam5Out.sd cam6Out.sd]';
                    X3 = [cam1Out.sd  camFake1.sd camFake2.sd camFake3.sd cam5Out.sd cam6Out.sd]';
                    
                    % Test for tracking failures & remove
                    badInds = find((max(X1,[],1) < 10) == 0);
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
                    
                elseif (strcmp(method,'simpleMouth'))
                    % simple vertical mouth using landmarks in the top
                    % middle and bottom middle of the mouth
              
                    d1 = norm(cam1(curRange,15) - cam1(curRange,19));
                    d2 = norm(cam2(curRange,15) - cam2(curRange,19));
                    d3 = norm(cam3(curRange,15) - cam3(curRange,19));
                    d4 = norm(cam4(curRange,15) - cam4(curRange,19));
                    d5 = norm(cam5(curRange,15) - cam5(curRange,19));
                    d6 = norm(cam6(curRange,15) - cam6(curRange,19));
                    d7 = norm(fake4(curRange,15) - fake4(curRange,19));
                    d8 = norm(fake3(curRange,15) - fake3(curRange,19));
                    d9 = norm(fake2(curRange,15) - fake2(curRange,19));
                    
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
                    
                else
                    % DWT directly on input with no PCA
                    
                    L = wmaxlev(size(cam1(curRange,:),2),'haar');
                    [c,~]  = wavedec2(cam1(curRange,:),L,'haar');
                    [c2,~] = wavedec2(cam2(curRange,:),L,'haar');
                    [c3,~] = wavedec2(cam3(curRange,:),L,'haar');
                    [c4,~] = wavedec2(cam4(curRange,:),L,'haar');
                    [c5,~] = wavedec2(cam5(curRange,:),L,'haar');
                    [c6,~] = wavedec2(cam6(curRange,:),L,'haar');
                    [c7,~] = wavedec2(fake4(curRange,:),L,'haar');
                    [c8,~] = wavedec2(fake3(curRange,:),L,'haar');
                    [c9,~] = wavedec2(fake2(curRange,:),L,'haar');

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

                end
                
                % Does the current window contain a faked frame? Check if we are in
                % the middle where both are real
                if isempty(intersect(curRange,(intervalWin + 1):(2*intervalWin)))
                    isFake = 1;
                else
                    isFake = 0;
                end
                
                % (1) TP if window contains a faked frame & fake is detected
                % (2) TN if window does not have fake & fake is not detected
                % (3) FP if window does not have fake & fake is detected
                % (4) FN if window contains a faked frame & fake is not detected
                % Assumption: whole set of fakes must be correct
                
                %%%%%%%%%%%%%%%%%%% 0 fakes case
                if (numFakes0 == 0)
                    acc0(2,startF) = 1; % TN
                else
                    acc0(3,startF) = 1; % FP
                end
                
                %%%%%%%%%%%%%%%%%%% 1 fake case
                % One fake detected
                if (numFakes1 == 1)
                    if (isFake == 0)
                        acc1(3,startF) = 1; % FP
                    else
                        % Check if detected fake is in correct spot
                        if (all(o1 == [1;1;1;2;1;1]) || all(o1 == [2;2;2;1;2;2]))
                            acc1(1,startF) = 1; % TP
                        else
                            acc1(4,startF) = 1; % FN-- fake detected, and there should be a fake, but we picked the wrong video
                        end
                    end
                    
                % More than one fake detected, but GT is only 1 fake, so only FP
                elseif (numFakes1 > 1)
                    acc1(3,startF) = 1; % FP
                    
                % No fakes detected
                else
                    if (isFake == 0)
                        acc1(2,startF) = 1; %TN
                    else
                        acc1(4,startF) = 1; %FN
                    end
                end
                
                %%%%%%%%%%%%%%%%%%% 2 fake case
                % Two fakes detected
                if (numFakes2 == 2)
                    if (isFake == 0)
                        acc2(3,startF) = 1; % FP
                    else
                        % Check if detected fake is in correct spot
                        if (all(o2 == [1;1;2;2;1;1]) || all(o2 == [2;2;1;1;2;2]))
                            acc2(1,startF) = 1; % TP
                        else
                            acc2(4,startF) = 1; % FN-- fake detected, and there should be a fake, but we picked the wrong video
                        end
                    end
                    
                % Different number of fakes detected, but GT is only 2 fakes, so only FP
                elseif (numFakes2 == 1) || (numFakes2 > 2)
                    acc2(3,startF) = 1; % FP
                    
                % No fakes detected
                else
                    if (isFake == 0)
                        acc2(2,startF) = 1; %TN
                    else
                        acc2(4,startF) = 1; %FN
                    end
                end
                
                %%%%%%%%%%%%%%%%%%% 3 fake case
                % Three fakes detected
                if (numFakes3 == 3)
                    if (isFake == 0)
                        acc3(3,startF) = 1; % FP
                    else
                        % Check if detected fake is in correct spot
                        if (all(o3 == [1;2;2;2;1;1]) || all(o3 == [2;1;1;1;2;2]))
                            acc3(1,startF) = 1; % TP
                        else
                            acc3(4,startF) = 1; % FN-- fake detected, and there should be a fake, but we picked the wrong video
                        end
                    end
                    
                % Different number of fakes detected, but GT is only 3 fakes, so only FP
                elseif (numFakes3 == 1) || (numFakes3 == 2) || (numFakes3 > 3)
                    acc3(3,startF) = 1; % FP
                    
                % No fakes detected
                else
                    if (isFake == 0)
                        acc3(2,startF) = 1; %TN
                    else
                        acc3(4,startF) = 1; %FN
                    end
                end
                
                base(startF) = mean(baseline(curRange));
                
            end
            
            % Save data for this window size
            parsave(['Output/ID' person '/thresh_' num2str(t) '/' method '_window_' num2str(i) '.mat'], acc0, acc1, acc2, acc3, base, thresh, person);
        end
        
    end

end

%% Helpers
function parsave(fname,acc0,acc1,acc2,acc3,base,thresh,person)
save(fname, 'acc0', 'acc1','acc2','acc3','base','thresh','person');
end
