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

    % Match descriptors between the two images
    [desc1, locs1, desc2, locs2, matchings, ~] = match('image1.pgm', 'image2.pgm');
    
    % Get matching points
    [loca1, loca2] = get_matching_pts(locs1, locs2, matchings);

    % Apply RANSAC to estimate homography
    [H, inliers] = ransacfithomography(loca1, loca2, 0.01);
    % Compute error with respect to GT homography
    error = compute_homography_error(H, GT_homography, loca1, loca2, inliers);
end
function error = compute_homography_error(H, GT_homography, loca1, loca2, inliers)
    % Adjust the function to work with points in [2 × numPoints] format
    % Get inlier points
    loca1_inliers = loca1(:, inliers);
    % Add homogeneous coordinate
    loca1_homogeneous = [loca1_inliers; ones(1, size(loca1_inliers, 2))];
    % Transform points using the estimated homography
    loca1_transformed = H * loca1_homogeneous;
    loca1_transformed = loca1_transformed ./ repmat(loca1_transformed(3, :), 3, 1);
    % Transform points using the ground truth homography
    loca1_GT_transformed = GT_homography * loca1_homogeneous;
    loca1_GT_transformed = loca1_GT_transformed ./ repmat(loca1_GT_transformed(3, :), 3, 1);
    % Compute the error as the average Euclidean distance between the transformed points
    error = mean(sqrt(sum((loca1_transformed(1:2, :) - loca1_GT_transformed(1:2, :)).^2, 1))); 
end

% Visualize matched keypoints
function visualize_matches(im1, im2, loca1, loca2)
    % Convert images to RGB if they are grayscale
    if size(im1, 3) == 1
        im1 = repmat(im1, [1, 1, 3]);
    end
    if size(im2, 3) == 1
        im2 = repmat(im2, [1, 1, 3]);
    end
    % Concatenate images side by side
    im_combined = [im1, im2];
    % Adjust loca2 coordinates for the concatenated image
    loca2(1, :) = loca2(1, :) + size(im1, 2);
    % Display the concatenated image
    figure;
    imshow(im_combined);
    hold on;
    % Plot matched keypoints
    for i = 1:size(loca1, 2)
        plot([loca1(1, i), loca2(1, i)], [loca1(2, i), loca2(2, i)], 'r-');
        plot(loca1(1, i), loca1(2, i), 'go');
        plot(loca2(1, i), loca2(2, i), 'bo');
    end
    hold off;
end







