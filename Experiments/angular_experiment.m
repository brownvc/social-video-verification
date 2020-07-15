% This experiment tests whether all sets of cameras separated by the same
% real-world angular distance are more similar to one another than to a
% fake video. The fake here is of camera 4, which is one of the mostly
% front-facing cameras relative to the participant.
%
% Written by Eleanor Tursman
% Last updated 7/2020

clearvars; close all;
fprintf('Angular Test \n');

% Iterate test over people
people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
for p=1:length(people)
    
    person = people{p};
    fprintf('Current person: %s\n', person);
    
    data4 = load(['Data/mouth-data-fake4-ID' person '.mat']);
    
    fullLen = length(data4.cam1);
    
    cam1 = data4.cam1(1:fullLen,:);
    cam2 = data4.cam2(1:fullLen,:);
    cam3 = data4.cam3(1:fullLen,:);
    cam4 = data4.cam4(1:fullLen,:);
    cam5 = data4.cam5(1:fullLen,:);
    cam6 = data4.cam6(1:fullLen,:);
    
    % Split into thirds: [fake | real | fake]
    intervalWin = floor(fullLen / 3);
    fake4 = [data4.fake(1:intervalWin,:); cam4((intervalWin + 1):(2*intervalWin),:); data4.fake((2*intervalWin + 1):fullLen,:)];
    
    % calculate L2 between real/fake mouth landmarks for all frames
    baseline = vecnorm(cam4 - fake4, 2, 2);
    
    % Iterate over thresholds too
    threshes = [1.1 1.3 1.5];
    parfor t=1:length(threshes)
        thresh = threshes(t);
        
        % Iterate over angles between cameras
        for a=1:5 %[65 55 44 33 22] are our angles, in degrees
            fprintf('Angle: %d \n',a);
            
            for i=[150 250]
                fprintf('Window size: %d\n',i);
                
                numWin = fullLen - i;
                
                % format: row 1- TP, row 2- TN, row 3- FP, row 4- FN
                acc1 = zeros(4,numWin);
                base = zeros(4,numWin);
                
                % Iterate through windows
                for startF=1:fullLen
                    
                    endF = startF + i;
                    curRange = startF:endF;
                    
                    % Check if we go over the max number of frames
                    if (endF > fullLen)
                        continue;
                    end
                    
                    % Pick correct pair of real cameras based on real world
                    % angular separation, where cameras 1 through 6 are in
                    % an arc facing the participant. 1 & 6 are 65 degrees
                    % apart, 1 & 5 and 2 & 6 are 55 degrees apart, etc.
                    if(a == 1)
                        % Pairs: (1,6)
                        X = {angularHelper(cam1(curRange,:),cam6(curRange,:),fake4(curRange,:))};
                    elseif (a == 2)
                        % Pairs: (1,5),(2,6)
                        X = {angularHelper(cam1(curRange,:),cam5(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam2(curRange,:),cam6(curRange,:),fake4(curRange,:))};
                    elseif (a == 3)
                        % Pairs: (1,4),(2,5),(3,6)
                        X = {angularHelper(cam1(curRange,:),cam4(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam2(curRange,:),cam5(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam3(curRange,:),cam6(curRange,:),fake4(curRange,:))};
                    elseif (a == 4)
                        % Pairs: (1,3),(2,4),(3,5),(4,6)
                        X = {angularHelper(cam1(curRange,:),cam3(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam2(curRange,:),cam4(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam3(curRange,:),cam5(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam4(curRange,:),cam6(curRange,:),fake4(curRange,:))};
                    else
                        % Pairs: (1,2),(2,3),(3,4),(4,5),(5,6)
                        X = {angularHelper(cam1(curRange,:),cam2(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam2(curRange,:),cam3(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam3(curRange,:),cam4(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam4(curRange,:),cam5(curRange,:),fake4(curRange,:)), ...
                            angularHelper(cam5(curRange,:),cam6(curRange,:),fake4(curRange,:))};
                    end
                    
                    % Iterate over pairs, and take an average result
                    for pair=1:length(X)
                        
                        curX = X{pair};
                        [~,tree] = clusterdata(curX,4);
                        [numFakes1,o1] = detectFakesTree(tree,thresh);
                        
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
                        %%%%%%%%%%%%%%%%%%% 1 fake case
                        % One fake detected
                        if (numFakes1 == 1)
                            if (isFake == 0)
                                acc1(3,startF) = acc1(3,startF) + 1; % FP
                            else
                                % Check if detected fake is in correct spot
                                if (all(o1 == [1;1;2]) || all(o1 == [2;2;1]))
                                    acc1(1,startF) = acc1(1,startF) + 1; % TP
                                else
                                    acc1(4,startF) = acc1(4,startF) + 1; % FN-- fake detected, and there should be a fake, but we picked the wrong video
                                end
                            end
                            
                        % More than one fake detected, but GT is only 1 fake, so only FP
                        elseif (numFakes1 > 1)
                            acc1(3,startF) = acc1(3,startF) + 1; % FP
                            
                        % No fakes detected
                        else
                            if (isFake == 0)
                                acc1(2,startF) = acc1(2,startF) + 1; %TN
                            else
                                acc1(4,startF) = acc1(4,startF) + 1; %FN
                            end
                        end
                        
                    end
                    
                    base(startF) = mean(baseline(curRange));
                    
                end
                
                % Take mean over results
                acc1 = acc1 ./ length(X);
                
                % Save data for this window size
                parsave(['Output/ID' person '/thresh_' num2str(t) '/angle_' num2str(a) '_window_' num2str(i) '.mat'], acc1, base, thresh, person, a);
                
            end
            
        end
    end
end
%% Helpers
function parsave(fname,acc1,base,thresh,person,a)
save(fname, 'acc1','base','thresh','person','a');
end
