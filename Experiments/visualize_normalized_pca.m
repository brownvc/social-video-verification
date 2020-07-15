% Visualization script to confirm pca results looks reasonably different
% for fake than for reals.
%
% Written by Eleanor Tursman
% Last updated 7/2020
clearvars; close all;

% Load input data
load('Data/mouth-data-fake3-ID7.mat');

cam1Out = cpca(cam1,'k',2);
cam2Out = cpca(cam2,'k',2);
cam3Out = cpca(cam3,'k',2);
cam4Out = cpca(cam4,'k',2);
cam5Out = cpca(cam5,'k',2);
cam6Out = cpca(cam6,'k',2);
camFakeOut = cpca(fake,'k',2);

figure('units','normalized','outerposition',[0 0 1 1]);

data = {cam1; cam2; cam3; cam4; cam5; cam6; fake};

for i=(1:size(cam1,1))
    sgtitle(['Frame ' num2str(i)]);
    
    subplot(2,4,1);
    cla;
    pts = reshape(data{1}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 1'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,2);
    cla;
    pts = reshape(data{2}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 2'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,3);
    cla;
    pts = reshape(data{3}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 3'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,4);
    cla;
    pts = reshape(data{4}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 4'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,5);
    cla;
    pts = reshape(data{5}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 5'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,6);
    cla;
    pts = reshape(data{6}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Camera 6'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    subplot(2,4,7);
    cla;
    pts = reshape(data{7}(i,:),[2,20])';
    scatter(pts(:,1),pts(:,2),10,'filled'); hold on;
    title('Fake'); hold on;
    xlim([-5 5]); hold on;
    ylim([-5 5]); hold on;
    
    pause(0.01);
    
end



