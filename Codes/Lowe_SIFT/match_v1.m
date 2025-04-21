function [desc1, loca1, desc2, loca2, matchings, mnb] = match(image1, image2)
    % Find SIFT keypoints for each image
    [im1, des1, loc1] = sift(image1);
    [im2, des2, loc2] = sift(image2);
    
    % Filter low-contrast keypoints
    contrast_threshold = 0.08; % Eski: 0.06
    valid_indices1 = loc1(:, 3) > contrast_threshold;
    loc1 = loc1(valid_indices1, :);
    des1 = des1(valid_indices1, :);
    valid_indices2 = loc2(:, 3) > contrast_threshold;
    loc2 = loc2(valid_indices2, :);
    des2 = des2(valid_indices2, :);
    
    % Match descriptors
    distRatio = 0.3; % Eski: 0.35
    des2t = des2';
    for i = 1:size(des1, 1)
        dotprods = des1(i, :) * des2t;
        [vals, indx] = sort(acos(dotprods));
        if vals(1) < distRatio * vals(2)
            match(i) = indx(1);
        else
            match(i) = 0;
        end
    end
    
    % Visualize matches
    im3 = appendimages(im1, im2);
    figure('Position', [10 10 size(im3, 2) size(im3, 1)]);
    colormap('gray');
    imagesc(im3);
    hold on;
    cols1 = size(im1, 2);
    for i = 1:size(des1, 1)
        if match(i) > 0
            line([loc1(i, 2) loc2(match(i), 2) + cols1], ...
                 [loc1(i, 1) loc2(match(i), 1)], 'Color', 'c');
        end
    end
    hold off;
    
    num = sum(match > 0);
    matchings = match;
    mnb = num;
    desc1 = des1;
    desc2 = des2;
    loca1 = loc1;
    loca2 = loc2;
    fprintf('Found %d matches.\n', num);
end