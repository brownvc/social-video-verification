% This script was used to generate most of the plots in the paper. 
%
% Written by Eleanor Tursman
% Last updated 7/2020

%% Make the plots
clearvars; close all;

histogramOn = false;
accOn = true;
prOn = false;
rocOn = false;
angle = false;

datasetName = 'onlyPCA';

%% angles
if(angle)
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    accResults = zeros(1,5,length(people)); % # angles, # people
    thresh = 2; % ind in [1.1 1.3 1.5]
    
    for p=1:length(people)
        
        
        fnameRoot = ['Output/ID' num2str(p) '/thresh_' num2str(thresh) '/'];
        
        % load the data-- the data was saved s.t. angle 1 was 65 degrees, 2
        % was 52, etc. We reverse it here for ease in plotting.
        ang_1 = load([fnameRoot datasetName '_5_window_250.mat']);
        ang_2 = load([fnameRoot datasetName '_4_window_250.mat']);
        ang_3 = load([fnameRoot datasetName '_3_window_250.mat']);
        ang_4 = load([fnameRoot datasetName '_2_window_250.mat']);
        ang_5 = load([fnameRoot datasetName '_1_window_250.mat']);
        
        dataset = {ang_1,ang_2,ang_3,ang_4,ang_5};
  
        
        numAng = length(dataset);
        accs = zeros(1,numAng);
        
        for i=1:numAng
            
            data = dataset{i};
            accs(1,i) = (sum(data.acc1(1,:)) + sum(data.acc1(2,:))) / (sum(data.acc1(1,:)) + sum(data.acc1(2,:)) + sum(data.acc1(3,:)) + sum(data.acc1(4,:)));
            
        end
        
        accResults(:,:,p) = accs;
        
    end
    
    
    % Plot average result per window + one stdev
    meanRes = mean(accResults,3);
    stdRes = std(accResults,0,3);
    
    accXData = [13 26 39 52 65];
    
    figure;
    errorbar(accXData,meanRes,stdRes,'LineWidth',1.0); hold on;
    ylim([0 1]);
    set(gca,'FontSize',20);
    xlabel('Pairwise Angular Distance (Degrees)','FontSize',20);
    ylabel('Accuracy','FontSize',20);
    title('Detection Accuracy vs Pairwise Angular Distance','FontSize',20);    
end

%% make histograms
if (histogramOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    thresh = 2; % ind in [1.1 1.3 1.5 1.7 1.9 2.1]
    bins = 0.1:0.1:20;
    
    hist50 = zeros(2,length(bins),length(people));
    hist150 = zeros(2,length(bins),length(people));
    hist250 = zeros(2,length(bins),length(people));
    hist350 = zeros(2,length(bins),length(people));
    
    for p=1:length(people)
        
        fnameRoot = ['Output/ID' num2str(p) '/thresh_' num2str(thresh) '/'];
        
        % load the data
        win50 = load([fnameRoot datasetName '_window_50.mat']);
        win150 = load([fnameRoot datasetName '_window_150.mat']);
        win250 = load([fnameRoot datasetName '_window_250.mat']);
        win350 = load([fnameRoot datasetName '_window_350.mat']);
        
        % Want to take fake sect1ions, TP row. TP - row 2, base - row 1
        interval = floor(length(win50.acc1) / 3);
        
        data50 =  [win50.base(2,1:interval) win50.base(2,2*interval + 1:end); ...
            win50.acc1(1,1:interval) win50.acc1(1,2*interval + 1:end)];
        data150 = [win150.base(2,1:interval) win150.base(2,2*interval + 1:end); ...
            win150.acc1(1,1:interval) win150.acc1(1,2*interval + 1:end)];
        data250 = [win250.base(2,1:interval) win250.base(2,2*interval + 1:end); ...
            win250.acc1(1,1:interval) win250.acc1(1,2*interval + 1:end)];
        data350 = [win350.base(2,1:interval) win350.base(2,2*interval + 1:end); ...
            win350.acc1(1,1:interval) win350.acc1(1,2*interval + 1:end)];
        
        % Sort by base
        sorted50 = sortrows(data50');
        sorted150 = sortrows(data150');
        sorted250 = sortrows(data250');
        sorted350 = sortrows(data350');
        
        % Iterate through pre-set bins
        hist50(:,:,p) = getBinCounts(sorted50,bins);
        hist150(:,:,p) = getBinCounts(sorted150,bins);
        hist250(:,:,p) = getBinCounts(sorted250,bins);
        hist350(:,:,p) = getBinCounts(sorted350,bins);
        
    end
    
    % Take avg and std
    mean50 = mean(hist50,3)';
    std50 = std(hist50,0,3)';
    mean150 = mean(hist150,3)';
    std150 = std(hist150,0,3)';
    mean250 = mean(hist250,3)';
    std250 = std(hist250,0,3)';
    mean350 = mean(hist350,3)';
    std350 = std(hist350,0,3)';
    
    figure('Position', [10 10 1500 500]);
    set(gca,'FontSize',20);
    subplot(1,4,1);
    bar(bins,mean50); hold on;
    legend('Undetected Fake in Window','Fake Detected in Window');
    ylabel('Count');
    xlabel('L2 Difference Between Real and Fake Mouth');
    title('One Fake, Thresh = 1.3, Window Size = 50');
    xlim([0 20]); ylim([0 25]);
    set(gca,'FontSize',20);
    
    subplot(1,4,2);
    bar(bins,mean150); hold on;
    legend('Undetected Fake in Window','Fake Detected in Window');
    ylabel('Count');
    xlabel('L2 Difference Between Real and Fake Mouth');
    title('One Fake, Thresh = 1.3, Window Size = 150');
    xlim([0 20]); ylim([0 25]);
    set(gca,'FontSize',20);
    
    subplot(1,4,3);
    bar(bins,mean250); hold on;
    legend('Undetected Fake in Window','Fake Detected in Window');
    ylabel('Count'); ylim([0 25]);
    xlabel('L2 Difference Between Real and Fake Mouth');
    title('One Fake, Thresh = 1.3, Window Size = 250');
    xlim([0 20]); ylim([0 25]);
    set(gca,'FontSize',20);
    
    subplot(1,4,4);
    bar(bins,mean350); hold on;
    legend('Undetected Fake in Window','Fake Detected in Window');
    ylabel('Count');
    xlabel('L2 Difference Between Real and Fake Mouth');
    title('One Fake, Thresh = 1.3, Window Size = 350');
    xlim([0 20]); ylim([0 25]);
    set(gca,'FontSize',20);
    
    
end

%% make accuracy plots
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% Accuracy = (TP+TN)/(TP+TN+FP+FN)
if (accOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    accResults = zeros(4,4,length(people));
    thresh = 2; % ind in [1.1 1.3 1.5 1.7 1.9 2.1]
    
    for p=1:length(people)
        
        
        fnameRoot = ['Output/ID' num2str(p) '/thresh_' num2str(thresh) '/'];
        
        % load the data
        win50 = load([fnameRoot datasetName '_window_50.mat']);
        win150 = load([fnameRoot datasetName '_window_150.mat']);
        win250 = load([fnameRoot datasetName '_window_250.mat']);
        win350 = load([fnameRoot datasetName '_window_350.mat']);
        
        dataset = {win50,win150,win250,win350};
        accXData = [50 150 250 350];
        
        numWin = length(dataset);
        accs = zeros(4,numWin);
        
        for i=1:numWin
            
            data = dataset{i};
            accs(1,i) = (sum(data.acc0(1,:)) + sum(data.acc0(2,:))) / (sum(data.acc0(1,:)) + sum(data.acc0(2,:)) + sum(data.acc0(3,:)) + sum(data.acc0(4,:)));
            accs(2,i) = (sum(data.acc1(1,:)) + sum(data.acc1(2,:))) / (sum(data.acc1(1,:)) + sum(data.acc1(2,:)) + sum(data.acc1(3,:)) + sum(data.acc1(4,:)));
            accs(3,i) = (sum(data.acc2(1,:)) + sum(data.acc2(2,:))) / (sum(data.acc2(1,:)) + sum(data.acc2(2,:)) + sum(data.acc2(3,:)) + sum(data.acc2(4,:)));
            accs(4,i) = (sum(data.acc3(1,:)) + sum(data.acc3(2,:))) / (sum(data.acc3(1,:)) + sum(data.acc3(2,:)) + sum(data.acc3(3,:)) + sum(data.acc3(4,:)));
            
        end
        
        accResults(:,:,p) = accs;
        
    end
    
    
    % Plot average result per window + one stdev
    meanRes = mean(accResults,3);
    stdRes = std(accResults,0,3);
    
    figure;
    errorbar(accXData,meanRes(1,:),stdRes(1,:)); hold on;
    errorbar(accXData,meanRes(2,:),stdRes(2,:)); hold on;
    errorbar(accXData,meanRes(3,:),stdRes(3,:)); hold on;
    errorbar(accXData,meanRes(4,:),stdRes(4,:)); hold on;
    ylim([0 1]);
    xlim([0 400]);
    xlabel('Window Size');
    ylabel('Accuracy');
    title('Detection Accuracy vs Window Size');
    set(gca,'FontSize',20);
    legend('No Fakes','One Fake','Two Fakes','Three Fakes');
    
end

%% precision recall
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% precision = tp / (tp + fp), recall = tp / (tp + fn)

if(prOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    threshNum = 6; % length of [1.1 1.3 1.5 1.7 1.9 2.1]
    win = 250; % [50 150 250 350]
    pResults = zeros(threshNum,3,length(people));
    rResults = zeros(threshNum,3,length(people));
    
    for p=1:length(people)
        
        fnameRoot = ['Output/ID' num2str(p) '/'];
        
        data1 = load([fnameRoot 'thresh_1/' datasetName '_window_' num2str(win) '.mat']);
        data2 = load([fnameRoot 'thresh_2/' datasetName '_window_' num2str(win) '.mat']);
        data3 = load([fnameRoot 'thresh_3/' datasetName '_window_' num2str(win) '.mat']);
        data4 = load([fnameRoot 'thresh_4/' datasetName '_window_' num2str(win) '.mat']);
        data5 = load([fnameRoot 'thresh_5/' datasetName '_window_' num2str(win) '.mat']);
        data6 = load([fnameRoot 'thresh_6/' datasetName '_window_' num2str(win) '.mat']);
        
        dataset = {data1,data2,data3,data4,data5,data6};
        
        for i=1:threshNum
            
            data = dataset{i};
            
            if (sum(data.acc1(1,:)) == 0) && ( (sum(data.acc1(1,:)) + sum(data.acc1(3,:))) == 0)
                pResults(i,1,p) = 1;
            else
                pResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(3,:)));
            end
            
            if (sum(data.acc2(1,:)) == 0) && ( (sum(data.acc2(1,:)) + sum(data.acc2(3,:))) == 0)
                pResults(i,2,p) = 1;
            else
                pResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(3,:)));
            end
            
            % Running into NaN errors for 0/0
            if (sum(data.acc3(1,:)) == 0) && ( (sum(data.acc3(1,:)) + sum(data.acc3(3,:))) == 0)
                pResults(i,3,p) = 1;
            else
                pResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(3,:)));
            end
            
            rResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(4,:)));
            rResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(4,:)));
            rResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(4,:)));
            
        end
        
    end
    
    % Aggregate results-- can't do std without taking multiple measurements
    % of -each- threshold
    meanP = mean(pResults,3);
    meanR = mean(rResults,3);
    stdP = std(pResults,0,3);
    stdR = std(rResults,0,3);
    
    % reformat & order by recall values
    oneFake = sortrows([meanR(:,1) meanP(:,1)]);
    twoFake = sortrows([meanR(:,2) meanP(:,2)]);
    thrFake = sortrows([meanR(:,3) meanP(:,3)]);
    
    figure;
    errorbar(oneFake(:,1),oneFake(:,2),stdP(:,1),stdP(:,1),stdR(:,1),stdR(:,1)); hold on;
    errorbar(twoFake(:,1),twoFake(:,2),stdP(:,2),stdP(:,2),stdR(:,2),stdR(:,2)); hold on;
    errorbar(thrFake(:,1),thrFake(:,2),stdP(:,3),stdP(:,3),stdR(:,3),stdR(:,3)); hold on;
    legend('One Fake','Two Fakes','Three Fakes');
    xlabel('Recall');
    ylabel('Precision');
    xlim([0 1]);
    ylim([0 1]);
    title('Precision v Recall, Window size = 250');
    
end

%% make roc curves
% (1) TP if window contains a faked frame & fake is detected
% (2) TN if window does not have fake & fake is not detected
% (3) FP if window does not have fake & fake is detected
% (4) FN if window contains a faked frame & fake is not detected
% roc: TPR vs FPR at various threshes
% TPR = tp / (tp + fn)
% FPR = fp / (fp + tn)
if (rocOn)
    
    people = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'};
    threshNum = 6; % length of [1.1 1.3 1.5 1.7 1.9 2.1]
    win = 350; % [50 150 250 350]
    tpResults = zeros(threshNum,3,length(people));
    fpResults = zeros(threshNum,3,length(people));
    fpZeroFake = zeros(threshNum,1,length(people));
    
    for p=1:length(people)
        
        fnameRoot = ['Output/ID' num2str(p) '/'];
        
        data1 = load([fnameRoot 'thresh_1/' datasetName '_window_' num2str(win) '.mat']);
        data2 = load([fnameRoot 'thresh_2/' datasetName '_window_' num2str(win) '.mat']);
        data3 = load([fnameRoot 'thresh_3/' datasetName '_window_' num2str(win) '.mat']);
        data4 = load([fnameRoot 'thresh_4/' datasetName '_window_' num2str(win) '.mat']);
        data5 = load([fnameRoot 'thresh_5/' datasetName '_window_' num2str(win) '.mat']);
        data6 = load([fnameRoot 'thresh_6/' datasetName '_window_' num2str(win) '.mat']);
        
        dataset = {data1,data2,data3,data4,data5,data6};
        
        for i=1:threshNum
            
            data = dataset{i};
            tpResults(i,1,p) = sum(data.acc1(1,:)) / (sum(data.acc1(1,:)) + sum(data.acc1(4,:)));
            tpResults(i,2,p) = sum(data.acc2(1,:)) / (sum(data.acc2(1,:)) + sum(data.acc2(4,:)));
            tpResults(i,3,p) = sum(data.acc3(1,:)) / (sum(data.acc3(1,:)) + sum(data.acc3(4,:)));
            
            fpResults(i,1,p) = sum(data.acc1(3,:)) / (sum(data.acc1(3,:)) + sum(data.acc1(2,:)));
            fpResults(i,2,p) = sum(data.acc2(3,:)) / (sum(data.acc2(3,:)) + sum(data.acc2(2,:)));
            fpResults(i,3,p) = sum(data.acc3(3,:)) / (sum(data.acc3(3,:)) + sum(data.acc3(2,:)));
            
            fpZeroFake(i,1,p) = sum(data.acc0(3,:)) / (sum(data.acc0(3,:)) + sum(data.acc0(2,:)));
            
        end
    end
    
    meanTP = mean(tpResults,3);
    meanFP = mean(fpResults,3)
    stdTP = std(tpResults,0,3);
    stdFP = std(fpResults,0,3)
    
    zeroFakeMean = mean(fpZeroFake,3)
    zeroFakeStd = std(fpZeroFake,0,3)
    
    % reformat & order by recall values
    oneFake = sortrows([meanFP(:,1) meanTP(:,1)]);
    twoFake = sortrows([meanFP(:,2) meanTP(:,2)]);
    thrFake = sortrows([meanFP(:,3) meanTP(:,3)]);
    
    figure;
    errorbar(oneFake(:,1),oneFake(:,2),stdTP(:,1),stdTP(:,1),stdFP(:,1),stdFP(:,1)); hold on;
    errorbar(twoFake(:,1),twoFake(:,2),stdTP(:,2),stdTP(:,2),stdFP(:,2),stdFP(:,2)); hold on;
    errorbar(thrFake(:,1),thrFake(:,2),stdTP(:,3),stdTP(:,3),stdFP(:,3),stdFP(:,3)); hold on;
    legend('One Fake','Two Fakes','Three Fakes');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    xlim([0 1]);
    ylim([0 1]);
    title('ROC Curve, Window size = 250');
    set(gca,'FontSize',20);
    
end
%% Helpers
function curHist = getBinCounts(data,bins)

curHist = [];
for bin=bins
    
    curMin = bin;
    curMax = bin + 0.1;
    
    binInds = find((data(:,1) < curMax) & (data(:,1) >= curMin));
    
    zeroes = find(data(binInds,2) == 0);
    ones = find(data(binInds,2) == 1);
    
    curHist(1,end+1) = length(zeroes);
    curHist(2,end) = length(ones);
    
end

end
