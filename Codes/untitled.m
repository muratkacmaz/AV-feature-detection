% Load the baseline image
baseline_img = imread('baseline.jpg');
if size(baseline_img, 3) == 3
    baseline_img = rgb2gray(baseline_img);
end
% Define scales and rotations
scales = 1.2:0.4:2;
rotations = -60:20:60;
% Initialize results storage
methods = {'FAST', 'SURF', 'ORB', 'BRISK', 'KAZE', 'AKAZE', 'SIFT', 'HARRIS'};
num_methods = length(methods);
num_matches = zeros(num_methods, length(scales), length(rotations));
computation_time = zeros(num_methods, length(scales), length(rotations));
% Loop through scales and rotations
for s = 1:length(scales)
    for r = 1:length(rotations)
        % Generate synthetic image
        tform = affine2d([scales(s)*cosd(rotations(r)) -scales(s)*sind(rotations(r)) 0; ...
                          scales(s)*sind(rotations(r))  scales(s)*cosd(rotations(r)) 0; ...
                          0 0 1]);
        synthetic_img = imwarp(baseline_img, tform, 'OutputView', imref2d(size(baseline_img)));
        % Loop through each method
        % Loop through each detection function
        for i = 1:size(detection_functions, 1)
            func_name = detection_functions{i, 1};
            tic; % Start timer
            % Switch case for each detection method
            switch func_name
                case 'FAST'
                    features = detectFASTFeatures(image);
                case 'SIFT'
                    features = detectSIFTFeatures(image, 'ContrastThreshold', 0.04, 'EdgeThreshold', 10, 'NumLayersInOctave', 3, 'Sigma', 1.6);
                case 'SURF'
                    features = detectSURFFeatures(image, 'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 6);
                case 'KAZE'
                    features = detectKAZEFeatures(image, 'Threshold', 0.001, 'NumOctaves', 4, 'NumScaleLevels', 4);
                case 'BRISK'
                    features = detectBRISKFeatures(image, 'MinContrast', 0.2, 'MinQuality', 0.1);
                case 'ORB'
                    features = detectORBFeatures(image, 'ScaleFactor', 1.2, 'NumLevels', 8);
                case 'HARRIS'
                    features = detectHarrisFeatures(image, 'MinQuality', 0.01, 'FilterSize', 5);
                case 'MSER'
                    features = detectMSERFeatures(image, 'ThresholdDelta', 2, 'RegionAreaRange', [30 14000], 'MaxAreaVariation', 0.25);
                otherwise
                    error('Unknown detection method: %s', func_name);
            end
            time_elapsed = toc; % End timer
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
    end
end
% Display results
for m = 1:num_methods
    fprintf('Method: %s\n', methods{m});
    for s = 1:length(scales)
        for r = 1:length(rotations)
            fprintf('Scale: %.1f, Rotation: %d, Matches: %d, Time: %.2f s\n', ...
                scales(s), rotations(r), num_matches(m, s, r), computation_time(m, s, r));
        end
    end
end

