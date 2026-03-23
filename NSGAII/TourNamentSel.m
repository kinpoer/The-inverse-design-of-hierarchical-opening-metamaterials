function p = TourNamentSel(pop)
% Tournament selection operator.
% Input:
%   pop - population array
% Output:
%   p   - selected individual
%
% Two individuals are randomly chosen from the population.
% The one with better rank (smaller) is selected.
% If ranks are equal, the one with larger crowding distance is selected.

n = numel(pop);
% Randomly select two distinct indices
idx = randperm(n, 2);
candidate1 = pop(idx(1));
candidate2 = pop(idx(2));

if candidate1.rank < candidate2.rank
    p = candidate1;
elseif candidate1.rank > candidate2.rank
    p = candidate2;
else  % same rank
    if candidate1.crowdingdistance > candidate2.crowdingdistance
        p = candidate1;
    else
        p = candidate2;
    end
end

end