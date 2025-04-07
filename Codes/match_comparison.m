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
image=imread('sunflower.jpg');   %   <---- You can use your own data

if (size(image,3))
    image=rgb2gray(image);
end

% Generating synthetc images with changes on scale, rotation and
% illumination
scale = 1.2;    %<------ To tune
rotation = 40;  %<------ To tune
image2 = imrotate(imresize(image,scale),rotation);


%Selecting Descriptor
b=input('Selecting descriptor method: FAST (F), SIFT (S), SURF (U), KAZE (K), BRISK (B), ORB (O), HARRIS (H) or MSER (M) \n','s');
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

tic
% Feature extraction and matching
[features1,validPts1] = extractFeatures(image,pts1);
[features2,validPts2] = extractFeatures(image2,pts2);
indexPairs = matchFeatures(features1,features2);
matched1  = validPts1(indexPairs(:,1));
matched2 = validPts2(indexPairs(:,2));
% Removing outliers
[tform, inlierIdx] = estimateGeometricTransform2D(matched2, matched1,'similarity');
inlier2  = matched2(inlierIdx, :);
inlier1   = matched1(inlierIdx, :);
t=toc;

% Visualizing matches...
color_data=colormap(jet(size(inlier1.Location,1)));
figure(2)
subplot(121)
imshow(image)
hold on
for i=1:size(inlier1.Location,1)
    plot(inlier1.Location(i,1),inlier1.Location(i,2),'*','Color',color_data(i,:))
end
hold off
subplot(122)
imshow(image2)
hold on
for i=1:size(inlier2.Location,1)
    plot(inlier2.Location(i,1),inlier2.Location(i,2),'*','Color',color_data(i,:))
end
hold off

disp(['The number of matches is: ' num2str(size(inlier1.Location,1)) ' and the computation time is ' num2str(t) ' seconds']);


