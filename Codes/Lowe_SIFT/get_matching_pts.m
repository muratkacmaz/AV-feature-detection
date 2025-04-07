function [pts1 pts2] = get_matching_pts( loc1, loc2, matchings )
%GET_MATCHING_PTS Summary of this function goes here
%   Detailed explanation goes here

%function that takes as input the locations of the keypoints found by SIFT
%and the result of the matching process.
%It returns two matrices of size 2XN
%corresponding to (x,y) coordinates in the left and right images that were
%matched and categorized as correct matchings
pts1 = zeros(2, sum(matchings>0)) ;
pts2 = zeros(2, sum(matchings>0)) ;

counter = 0;
for i = 1:size(matchings,2)
    if matchings(i) > 0
        counter = counter +1;
        pts1(1,counter) = loc1(i,2);
        pts1(2,counter) = loc1(i,1);
        pts2(1,counter) = loc2(matchings(i),2);
        pts2(2,counter) = loc2(matchings(i),1);
    end
end




end

