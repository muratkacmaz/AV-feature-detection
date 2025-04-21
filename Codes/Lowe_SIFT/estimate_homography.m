function [H, error] = estimate_homography(im1, im2, GT_homography)
    addpath('../lib/');
    addpath('../Lowe_SIFT/');
    
    % Convert images to grayscale if they are not already
    if size(im1, 3) == 3
        im1 = rgb2gray(im1);
    end
    if size(im2, 3) == 3
        im2 = rgb2gray(im2);
    end
    
    % Enhance contrast
    im1 = imadjust(im1);
    im2 = imadjust(im2);
    
    % Save images as PGM format for SIFT processing
    imwrite(im1, 'image1.pgm');
    imwrite(im2, 'image2.pgm');

    % Match descriptors between the two images
    [desc1, locs1, desc2, locs2, matchings, num] = match('image1.pgm', 'image2.pgm');
    fprintf('Number of SIFT matches: %d\n', num);
    
    % Limit to top 50 matches
    distances = zeros(1, sum(matchings > 0));
    counter = 1;
    for i = 1:size(desc1, 1)
        if matchings(i) > 0
            dotprod = desc1(i, :) * desc2(matchings(i), :)';
            distances(counter) = acos(dotprod);
            counter = counter + 1;
        end
    end
    [~, sorted_indices] = sort(distances);
    max_matches = 50;
    valid_indices = sorted_indices(1:min(max_matches, length(sorted_indices)));
    valid_matchings = zeros(1, size(desc1, 1));
    counter = 1;
    for i = 1:size(desc1, 1)
        if matchings(i) > 0
            if ismember(counter, valid_indices)
                valid_matchings(i) = matchings(i);
            end
            counter = counter + 1;
        end
    end
    [loca1, loca2] = get_matching_pts(locs1, locs2, valid_matchings);

    % Apply RANSAC to estimate homography
    [H, inliers] = ransacfithomography(loca1, loca2, 0.001);
    fprintf('Number of inliers: %d\n', length(inliers));
    
    % Check inlier distribution
    if length(inliers) > 0
        inlier_pts1 = loca1(:, inliers);
        std_x = std(inlier_pts1(1, :));
        std_y = std(inlier_pts1(2, :));
        fprintf('Inliers x std: %.2f, y std: %.2f\n', std_x, std_y);
        if std_x < size(im1, 2)/10 || std_y < size(im1, 1)/10
            warning('Warning Inliers.');
        end
    end
    
    % Visualize inlier matches
    visualize_inlier_matches(im1, im2, loca1, loca2, inliers);
    
    % Compute error with respect to GT homography
    error = compute_homography_error(H, GT_homography, loca1, loca2, inliers, im1);
    
    % Clean up temporary files
    delete('image1.pgm');
    delete('image2.pgm');
end
function error = compute_homography_error(H, GT_homography, loca1, loca2, inliers, im1)
    % Normalize homography matrices
    H = H / H(3,3);
    GT_homography = GT_homography / GT_homography(3,3);
    
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
    
    % Compute the error as the average Euclidean distance
    error = mean(sqrt(sum((loca1_transformed(1:2, :) - loca1_GT_transformed(1:2, :)).^2, 1)));
    
    % Normalize error by image diagonal
    diagonal = sqrt(size(im1, 1)^2 + size(im1, 2)^2);
    normalized_error = error / diagonal;
    fprintf('Error: %.4f pixels, Normalized error: %.4f (%% of diagonal)\n', error, normalized_error);
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






function visualize_inlier_matches(im1, im2, loca1, loca2, inliers)
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
    loca2_adjusted = loca2;
    loca2_adjusted(1, :) = loca2(1, :) + size(im1, 2);
    
    % Display the concatenated image
    figure;
    imshow(im_combined);
    hold on;
    
    % Plot only inlier matched keypoints
    for i = 1:length(inliers)
        idx = inliers(i); 
        plot([loca1(1, idx), loca2_adjusted(1, idx)], [loca1(2, idx), loca2_adjusted(2, idx)], 'r-');
        plot(loca1(1, idx), loca1(2, idx), 'go'); 
        plot(loca2_adjusted(1, idx), loca2_adjusted(2, idx), 'bo'); 
    end
    
    title('Inlier Matches after RANSAC');
    hold off;
end