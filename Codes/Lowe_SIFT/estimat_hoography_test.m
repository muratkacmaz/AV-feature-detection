
im1 = imread('scene.pgm');
im2 = imread('basmati.pgm');
GT_homography = load('../data_set/bikes/H1to3p');

[H, error] = estimate_homography(im1, im2, GT_homography);

disp('Estimated Homography:');
disp(H);
disp('Error:');
disp(error);