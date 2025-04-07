function [H, error] = estimate_homography(im1, im2, GT_homography)
    % Add necessary paths
    addpath('../lib/');
    addpath('../Lowe_SIFT/');
    % Convert images to grayscale if they are not already
    if size(im1, 3) == 3
        im1 = rgb2gray(im1);
    end
    if size(im2, 3) == 3
        im2 = rgb2gray(im2);
    end
    % Save images as PGM format for SIFT processing
    imwrite(im1, 'image1.pgm');
    imwrite(im2, 'image2.pgm');
    % Compute SIFT descriptors
    [~, descrips1, locs1] = sift('image1.pgm');
    [~, descrips2, locs2] = sift('image2.pgm');
    % Match descriptors between the two images
    matchings = match('image1.pgm', 'image2.pgm');
    % Get matching points
    [loca1, loca2] = get_matching_pts(locs1, locs2, matchings);
    % Apply RANSAC to estimate homography
    [H, inliers] = ransacfithomography(loca1', loca2', 0.01);
    % Compute error with respect to GT homography
    error = compute_homography_error(H, GT_homography, loca1, loca2, inliers);
end
function error = compute_homography_error(H, GT_homography, loca1, loca2, inliers)
    % Transform points using the estimated homography
    loca1_transformed = H * [loca1(inliers, :) ones(length(inliers), 1)]';
    loca1_transformed = loca1_transformed ./ loca1_transformed(3, :);
    % Transform points using the ground truth homography
    loca1_GT_transformed = GT_homography * [loca1(inliers, :) ones(length(inliers), 1)]';
    loca1_GT_transformed = loca1_GT_transformed ./ loca1_GT_transformed(3, :);
    % Compute the error as the average Euclidean distance between the transformed points
    error = mean(sqrt(sum((loca1_transformed(1:2, :) - loca1_GT_transformed(1:2, :)).^2, 1)));
end