%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       LABORATORY 1
%%%       COMPUTER VISION 2024-2025
%%%       FEATURE DETECTION AND COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
close all
addpath data

% Load input image
image = imread('sunflower.jpg');
if (size(image,3))
    image = rgb2gray(image);
end

methods     = {'FAST','SIFT','SURF','KAZE','BRISK','ORB','HARRIS','MSER'};
numMethods  = numel(methods);
times       = zeros(numMethods,1);
numFeatures = zeros(numMethods,1);

% 1) FAST
tic;
features_fast = detectFASTFeatures(image);
times(1)       = toc;
numFeatures(1) = features_fast.Count;

% 2) SIFT
tic;
features_sift = detectSIFTFeatures(image, ...
    'ContrastThreshold',   0.04, ...
    'EdgeThreshold',       10,   ...
    'NumLayersInOctave',   4,    ...
    'Sigma',               1.6);
times(2)       = toc;
numFeatures(2) = features_sift.Count;

% 3) SURF
tic;
features_surf = detectSURFFeatures(image, ...
    'MetricThreshold',  1000,  ...
    'NumOctaves',       4,     ...
    'NumScaleLevels',   6);
times(3)       = toc;
numFeatures(3) = features_surf.Count;

% 4) KAZE
tic;
features_kaze = detectKAZEFeatures(image, ...
    'Threshold',       0.001, ...
    'NumOctaves',      4,     ...
    'NumScaleLevels',  4);
times(4)       = toc;
numFeatures(4) = features_kaze.Count;

% 5) BRISK
tic;
features_brisk = detectBRISKFeatures(image, ...
    'MinContrast',    0.2, ...
    'MinQuality',     0.1);
times(5)       = toc;
numFeatures(5) = features_brisk.Count;

% 6) ORB
tic;
features_orb = detectORBFeatures(image, ...
    'ScaleFactor',     1.2,  ...
    'NumLevels',       8);
times(6)       = toc;
numFeatures(6) = features_orb.Count;

% 7) HARRIS
tic;
features_harris = detectHarrisFeatures(image, ...
    'MinQuality',     0.01, ...
    'FilterSize',     5);
times(7)       = toc;
numFeatures(7) = features_harris.Count;

% 8) MSER
tic;
features_mser = detectMSERFeatures(image, ...
    'ThresholdDelta',     2, ...
    'RegionAreaRange', [30 14000], ...
    'MaxAreaVariation',  0.25);
times(8)       = toc;
numFeatures(8) = numel(features_mser);

% Build and display results table
T = table(methods(:), numFeatures, times, ...
    'VariableNames',{'Method','Count','Time_s'});
disp(T);

% Visualization
figure(1)
subplot(241)
imshow(image)
hold on
plot(features_fast.Location(:,1),features_fast.Location(:,2),'*r','MarkerSize',4)
hold off
title('FAST')

subplot(242)
imshow(image)
hold on
plot(features_sift.Location(:,1),features_sift.Location(:,2),'*r','MarkerSize',4)
hold off
title('SIFT')

subplot(243)
imshow(image)
hold on
plot(features_surf.Location(:,1),features_surf.Location(:,2),'*r','MarkerSize',4)
hold off
title('SURF')

subplot(244)
imshow(image)
hold on
plot(features_kaze.Location(:,1),features_kaze.Location(:,2),'*r','MarkerSize',4)
hold off
title('KAZE')

subplot(245)
imshow(image)
hold on
plot(features_brisk.Location(:,1),features_brisk.Location(:,2),'*r','MarkerSize',4)
hold off
title('BRISK')

subplot(246)
imshow(image)
hold on
plot(features_orb.Location(:,1),features_orb.Location(:,2),'*r','MarkerSize',4)
hold off
title('ORB')

subplot(247)
imshow(image)
hold on
plot(features_harris.Location(:,1),features_harris.Location(:,2),'*r','MarkerSize',4)
hold off
title('HARRIS')

subplot(248)
imshow(image)
hold on
plot(features_mser.Location(:,1),features_mser.Location(:,2),'*r','MarkerSize',4)
hold off
title('MSER')
