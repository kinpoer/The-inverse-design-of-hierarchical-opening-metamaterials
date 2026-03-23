function [y1, y2] = CrossOver(x1, x2)
% Crossover operator (multi-point random exchange).
% Input:
%   x1, x2 - 1xn parent vectors
% Output:
%   y1, y2 - 1xn offspring vectors
%
% A random subset of positions is selected (size between 1 and n-1)
% and the values at those positions are swapped between parents.

n = numel(x1);
% Randomly choose how many positions to swap (at least 1, at most n-1)
r = randi([1, n-1], 1);
% Randomly select r distinct indices
swap_idx = randperm(n, r);

% Offspring are copies of parents initially
y1 = x1;
y2 = x2;

% Swap selected positions
y1(swap_idx) = x2(swap_idx);
y2(swap_idx) = x1(swap_idx);

end