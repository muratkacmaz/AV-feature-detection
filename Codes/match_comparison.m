%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              LABORATORY 2
%%%              COMPUTER VISION 2024-2025
%%%              FEATURE DETECTION AND DESCRIPTION. MATCHING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
close all

addpath data_set

% Load input image
image = imread('sunflower.jpg');   % <---- You can use your own data
if size(image,3)
    image = rgb2gray(image);
end

% Selecting Descriptor
b = input('Selecting descriptor method: FAST (F), SIFT (S), SURF (U), KAZE (K), BRISK (B), ORB (O), HARRIS (H) or MSER (M) \n','s');

% Define scale and rotation ranges
scaleList = 1.2:0.4:2.0;
rotationList = -60:20:60;

% Prepare result storage
results = [];

for scale = scaleList
    for rotation = rotationList
        fprintf('Processing scale = %.1f, rotation = %d...\n', scale, rotation);

        % Generate synthetic image
        image2 = imrotate(imresize(image, scale), rotation);

        % Detect features based on user input
        try
            if (b=='F' || b=='f')
                pts1  = detectFASTFeatures(image);
                pts2 = detectFASTFeatures(image2);
            elseif (b=='S' || b=='s')
                pts1  = detectSIFTFeatures(image);
                pts2 = detectSIFTFeatures(image2);
            elseif (b=='U' || b=='u')
                pts1  = detectSURFFeatures(image);
                pts2 = detectSURFFeatures(image2);
            elseif (b=='K' || b=='k')
                pts1  = detectKAZEFeatures(image);
                pts2 = detectKAZEFeatures(image2);
            elseif (b=='B' || b=='b')
                pts1  = detectBRISKFeatures(image);
                pts2 = detectBRISKFeatures(image2);
            elseif (b=='O' || b=='o')
                pts1  = detectORBFeatures(image);
                pts2 = detectORBFeatures(image2);
            elseif (b=='H' || b=='h')
                pts1  = detectHarrisFeatures(image);
                pts2 = detectHarrisFeatures(image2);
            elseif (b=='M' || b=='m')
                pts1  = detectMSERFeatures(image);
                pts2 = detectMSERFeatures(image2);
            else
                error('The selection is incorrect');
            end

            % Feature extraction and matching
            tic
            [features1,validPts1] = extractFeatures(image, pts1);
            [features2,validPts2] = extractFeatures(image2, pts2);
            indexPairs = matchFeatures(features1, features2);

            matched1 = validPts1(indexPairs(:,1));
            matched2 = validPts2(indexPairs(:,2));

            if size(matched1,1) >= 3
                [~, inlierIdx] = estimateGeometricTransform2D(matched2, matched1, 'similarity');
                numMatches = length(inlierIdx);
            else
                numMatches = 0;
            end

            t = toc;
        catch
            numMatches = 0;
            t = NaN;
        end

        % Store result
        results = [results; {scale, rotation, numMatches, t}];
        if numMatches > 0
            % Görselleştirme
            color_data = colormap(jet(numMatches));
        
            figure;
            subplot(121);
            imshow(image);
            hold on;
            locs1 = matched1(inlierIdx).Location;
            numInliers = size(locs1, 1);
            color_data = colormap(jet(numInliers));
            
            for i = 1:numInliers
                plot(locs1(i,1), locs1(i,2), '*', 'Color', color_data(i,:));
            end
            title(sprintf('Original Image - Inliers (%d)', numMatches));
            hold off;
        
            subplot(122);
            imshow(image2);
            hold on;
            locs2 = matched2(inlierIdx).Location;
            numInliers = size(locs2, 1);
            color_data = colormap(jet(numInliers));
            
            for i = 1:numInliers
                plot(locs2(i,1), locs2(i,2), '*', 'Color', color_data(i,:));
            end
            title(sprintf('Transformed Image - Inliers (%d)', numMatches));
            hold off;
        else
            % Eşleşme yoksa boş subplot göster
            figure;
            subplot(121); imshow(image); title('No matches');
            subplot(122); imshow(image2); title('No matches');
        end
        
        % Bilgi yazdır
        disp(['Scale: ' num2str(scale) ', Rotation: ' num2str(rotation) ', Descriptor: ' upper(b)]);
        disp(['The number of matches is: ' num2str(numMatches) ' and the computation time is ' num2str(t) ' seconds']);

    end
end

% Convert and show results
resultsTable = cell2table(results, ...
    'VariableNames', {'Scale','Rotation','NumMatches','TimeSeconds'});

disp(resultsTable)
