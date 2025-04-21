%clear workspace before the run

clear all
clc
close all
clearvars

im1 = imread('../data_set/bikes/img1.ppm');
im2 = imread('../data_set/bikes/img3.ppm');
GT_homography = load('../data_set/bikes/H1to3p');

[H, error] = estimate_homography(im1, im2, GT_homography);

disp('Estimated Homography:');
disp(H);
disp('Error:');
disp(error);


