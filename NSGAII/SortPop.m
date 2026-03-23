function pop = SortPop(pop)
% Sort population by rank (ascending) and then by crowding distance (descending).
% Input:
%   pop - population array with fields .rank and .crowdingdistance
% Output:
%   pop - sorted population
%
% Note: The original code first sorts by crowding distance descending,
% then by rank ascending. This yields a valid but non-standard ordering.
% We preserve the original behavior to maintain compatibility.

% First sort by crowding distance descending
[~, idx_dist] = sort([pop.crowdingdistance], 'descend');
pop = pop(idx_dist);

% Then sort by rank ascending
[~, idx_rank] = sort([pop.rank]);
pop = pop(idx_rank);

end