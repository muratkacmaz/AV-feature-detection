function [pts1, pts2] = get_matching_pts(locs1, locs2, matchings)
    % This function extracts the matching points from SIFT locations and matching data
    % Check the structure of matchings
    if isstruct(matchings)
        % Handle case where matchings is a struct with match info
        numMatches = size(matchings, 1);
        pts1 = zeros(numMatches, 2);
        pts2 = zeros(numMatches, 2);
        for i = 1:numMatches
            idx1 = matchings(i).match(1);
            idx2 = matchings(i).match(2);
            pts1(i, :) = locs1(idx1, 1:2);
            pts2(i, :) = locs2(idx2, 1:2);
        end
    elseif ismatrix(matchings)
        % Handle case where matchings is a matrix
        valid_matches = find(matchings > 0);
        numMatches = length(valid_matches);
        pts1 = zeros(numMatches, 2);
        pts2 = zeros(numMatches, 2);
        for i = 1:numMatches
            idx1 = valid_matches(i);
            idx2 = matchings(idx1);
            pts1(i, :) = locs1(idx1, 1:2);
            pts2(i, :) = locs2(idx2, 1:2);
        end
    else
        error('Unexpected format for matchings variable');
    end
end