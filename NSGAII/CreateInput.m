function x = CreateInput(var)
% Create a random feasible input vector based on variable bounds and step sizes.
% Input:
%   var - 4 x n matrix where:
%         var(1,:) = lower bounds
%         var(2,:) = step sizes
%         var(3,:) = upper bounds
%         var(4,:) = number of steps (integer)
% Output:
%   x   - 1 x n random feasible vector

n = size(var, 2);
x = zeros(1, n);
for i = 1:n
    % Random integer between 0 and var(4,i) inclusive
    step_idx = randi([0, var(4, i)]);
    x(i) = var(1, i) + step_idx * var(2, i);
end

end