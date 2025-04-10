%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              LABORATORY #2 
%%%              COMPUTER VISION 2024-2025
%%%              CREATING IMAGE MOSAICS based on SIFT features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

addpath('../lib/');
addpath('../Lowe_SIFT/');

% INPUT DATA: reading pictures
I1=imread('picture1.jpg'); % <-- Introduce your picture 1
I2=imread('picture2.jpg'); % <-- Introduce your picture 2
imwrite(I1,'image1.pgm');
imwrite(I2,'image2.pgm');


  %%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% MISSING CODE HERE; compute matches based on SIFT
	%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SIFT matching
[~, loca1, ~, loca2, matchings, ~] = match('image1.pgm', 'image2.pgm');    

idx1 = find(matchings);
idx2 = matchings(idx1);
loca1 = [loca1(idx1,2),loca1(idx1,1)];
loca2 = [loca2(idx2,2),loca2(idx2,1)];


  %%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%% MISSING CODE HERE; compute the Homography H, and the inliers
	%%%%%%%%%%%%%%%%%%%%%%%%%%%
 




% Transpose to match expected input of ransacfithomography (2 x N)
[H, inliers] = ransacfithomography(loca2', loca1', 0.01);

tform = maketform('projective',H');
I21 = imtransform(I2,tform); 
figure(2),imshow(I1)
figure(3),imshow(I21)

% adjust color or grayscale linearly, using corresponding infomation
[M1, N1, dim] = size(I1);
[M2, N2, ~] = size(I2);
radius = 2;
x1ctrl = loca1(inliers,1);
y1ctrl = loca1(inliers,2);
x2ctrl = loca2(inliers,1);
y2ctrl = loca2(inliers,2);
ctrlLen = length(inliers);
s1 = zeros(1,ctrlLen);
s2 = zeros(1,ctrlLen);
for color = 1:dim
		for p = 1:ctrlLen
			left = round(max(1,x1ctrl(p)-radius));
			right = round(min(N1,left+radius+1));
			up = round(max(1,y1ctrl(p)-radius));
			down = round(min(M1,up+radius+1));
			s1(p) = sum(sum(I1(up:down,left:right,color))); 
		end
		for p = 1:ctrlLen
			left = round(max(1,x2ctrl(p)-radius));
			right = round(min(N2,left+radius+1));
			up = round(max(1,y2ctrl(p)-radius));
			down = round(min(M2,up+radius+1));
			s2(p) = sum(sum(I2(up:down,left:right,color)));
		end
		sc = (radius*2+1)^2*ctrlLen;
		adjcoef = polyfit(s1/sc,s2/sc,1);
		I1(:,:,color) = I1(:,:,color)*adjcoef(1)+adjcoef(2);
end


% Building the image mosaic
pt = zeros(3,4);
pt(:,1) = H*[1;1;1];
pt(:,2) = H*[N2;1;1];
pt(:,3) = H*[N2;M2;1];
pt(:,4) = H*[1;M2;1];
x2 = pt(1,:)./pt(3,:);
y2 = pt(2,:)./pt(3,:);

up = round(min(y2));
Y_offset = 0;
if up <= 0
	Y_offset = -up+1;
	up = 1;
end

left = round(min(x2));
X_offset = 0;
if left<=0
	X_offset = -left+1;
	left = 1;
end

% Showing results
[M3, N3, ~] = size(I21);
I_output(up:up+M3-1,left:left+N3-1,:) = I21;
I_output(Y_offset+1:Y_offset+M1,X_offset+1:X_offset+N1,:) = I1;
figure(10),imshow(I_output)