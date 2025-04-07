% Add necessary paths
addpath('../lib/');
addpath('../data_set/bikes/');
addpath('../data_set/boat/');
addpath('../data_set/graf/');
addpath('../data_set/leuven/');

% List of sub-folders
sub_folders = {'bikes', 'boat', 'graf', 'leuven'};

% Initialize results table
results = [];

% Loop through each sub-folder
for folder_idx = 1:length(sub_folders)
    folder = sub_folders{folder_idx};
    image_files = dir(fullfile('../data_set', folder, '*.ppm'));
    gt_files = dir(fullfile('../data_set', folder, 'H*'));

    if isempty(image_files)
        warning('No .ppm files found in folder: %s', folder);
        continue;
    end

    % Read the first image
    im1 = imread(fullfile('../data_set', folder, image_files(1).name));

    % Loop through each image in the sub-folder
    for img_idx = 2:length(image_files)
        im2 = imread(fullfile('../data_set', folder, image_files(img_idx).name));
        gt_homography = load(fullfile('../data_set', folder, gt_files(img_idx-1).name));

        % Estimate homography using the implemented function
        [H_est, error] = estimate_homography(im1, im2, gt_homography);

        % Apply the estimated homography to the first image
        im1_trans_est = imTrans(im1, H_est);

        % Apply the ground truth homography to the first image
        im1_trans_gt = imTrans(im1, gt_homography);

        % Show both transformed images side by side for comparison
        figure;
        subplot(1, 2, 1);
        imshow(im1_trans_est);
        title('Estimated Homography');
        subplot(1, 2, 2);
        imshow(im1_trans_gt);
        title('Ground Truth Homography');

        % Save the figure for the report
        saveas(gcf, fullfile('results', sprintf('%s_comparison_%d.png', folder, img_idx)));

        % Store the error in the results table
        results = [results; {folder, image_files(1).name, image_files(img_idx).name, error}];
    end
end

% Display results table
results_table = cell2table(results, 'VariableNames', {'SubFolder', 'Image1', 'Image2', 'Error'});
disp(results_table);

% Save results table to a file
writetable(results_table, 'results/errors.csv');