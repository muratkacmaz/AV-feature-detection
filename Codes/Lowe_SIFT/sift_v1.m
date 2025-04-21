function [image, descriptors, locs] = sift(imageFile)
    % Load image
    image = imread(imageFile);
    
    % Convert to grayscale if RGB
    [a, b, c] = size(image);
    if c == 3
        image = 0.2989*image(:, :, 1) + 0.587*image(:, :, 2) + 0.114*image(:, :, 3);
    end
    
    [rows, cols] = size(image); 
    
    % Convert into PGM imagefile
    f = fopen('tmp.pgm', 'w');
    if f == -1
        error('Could not create file tmp.pgm.');
    end
    fprintf(f, 'P5\n%d\n%d\n255\n', cols, rows);
    fwrite(f, image', 'uint8');
    fclose(f);
    
    % Call keypoints executable
    if isunix
        command = '!./sift ';
    else
        command = '!siftWin32 ';
    end
    command = [command ' <tmp.pgm >tmp.key'];
    eval(command);
    
    % Open tmp.key and check its header
    g = fopen('tmp.key', 'r');
    if g == -1
        error('Could not open file tmp.key.');
    end
    [header, count] = fscanf(g, '%d %d', [1 2]);
    if count ~= 2
        error('Invalid keypoint file beginning.');
    end
    num = header(1);
    len = header(2);
    if len ~= 128
        error('Keypoint descriptor length invalid (should be 128).');
    end
    
    % Creates the two output matrices
    locs = double(zeros(num, 4));
    descriptors = double(zeros(num, 128));
    
    % Parse tmp.key
    for i = 1:num
        [vector, count] = fscanf(g, '%f %f %f %f', [1 4]); % row col scale ori
        if count ~= 4
            error('Invalid keypoint file format');
        end
        locs(i, :) = vector(1, :);
        
        [descrip, count] = fscanf(g, '%d', [1 len]);
        if count ~= 128
            error('Invalid keypoint file value.');
        end
        descrip = descrip / sqrt(sum(descrip.^2));
        descriptors(i, :) = descrip(1, :);
    end
    fclose(g);
    
    % Filter low-contrast keypoints
    contrast_threshold = 0.08; % Eski: 0.06
    valid_indices = locs(:, 3) > contrast_threshold;
    locs = locs(valid_indices, :);
    descriptors = descriptors(valid_indices, :);
    
    % Limit to top 300 features
    max_features = 300; % Eski: 500
    [~, sorted_idx] = sort(locs(:, 3), 'descend');
    locs = locs(sorted_idx(1:min(max_features, size(locs, 1))), :);
    descriptors = descriptors(sorted_idx(1:min(max_features, size(locs, 1))), :);
end