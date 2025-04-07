%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              LABORATORY 1
%%%              COMPUTER VISION 2024-2025
%%%              FEATURE DETECTION AND COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

addpath data

% Load input image
image = imread('sunflower.jpg');
if (size(image, 3))
    image = rgb2gray(image);
end

% Initialize results table
results = [];

% Define detection functions and their parameters
detection_functions = {
    'FAST', @detectFASTFeatures, {};
    'SIFT', @detectSIFTFeatures, {'ContrastThreshold', 0.04, 'EdgeThreshold', 10, 'NumLayersInOctave', 3, 'Sigma', 1.6};
    'SURF', @detectSURFFeatures, {'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 6};
    'KAZE', @detectKAZEFeatures, {'Threshold', 0.001, 'NumOctaves', 4, 'NumScaleLevels', 4};
    'BRISK', @detectBRISKFeatures, {'MinContrast', 0.2, 'MinQuality', 0.1};
    'ORB', @detectORBFeatures, {'ScaleFactor', 1.2, 'NumLevels', 8};
    'HARRIS', @detectHarrisFeatures, {'MinQuality', 0.01, 'FilterSize', 5};
    'MSER', @detectMSERFeatures, {'ThresholdDelta', 2, 'RegionAreaRange', [30 14000], 'MaxAreaVariation', 0.25};
};

% Loop through each detection function
for i = 1:size(detection_functions, 1)
    func_name = detection_functions{i, 1};
    func_handle = detection_functions{i, 2};
    func_params = detection_functions{i, 3};
    
    % Measure computation time
    tic;
    features = func_handle(image, func_params{:});
    time_elapsed = toc;
    
    % Store results
    results = [results; {func_name, length(features), time_elapsed}];
    
    % Visualize detected features
    figure;
    imshow(image);
    hold on;
    plot(features.Location(:, 1), features.Location(:, 2), '*r', 'MarkerSize', 4);
    hold off;
    title(func_name);
end

% Display results table
results_table = cell2table(results, 'VariableNames', {'Algorithm', 'NumFeatures', 'TimeElapsed'});
disp(results_table);