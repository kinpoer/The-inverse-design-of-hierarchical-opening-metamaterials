function pop = CrowdingDistance(pop, F)
% Compute crowding distance for each individual in the population.
% Input:
%   pop - population array (struct with fields .cost, .crowdingdistance, etc.)
%   F   - cell array where F{i} contains indices of individuals in front i
% Output:
%   pop - updated population with .crowdingdistance field filled
%
% For each front, individuals are sorted by each objective and the
% crowding distance is computed as the sum of distances to neighbors.

n_fronts = numel(F);

for k = 1:n_fronts
    idx = F{k};                     % indices of individuals in this front
    n = numel(idx);
    if n == 0
        continue;
    end
    costs = [pop(idx).cost];        % nobj x n matrix
    nobj = size(costs, 1);
    
    % Initialize distance matrix (n individuals x nobj objectives)
    dist = zeros(n, nobj);
    
    for j = 1:nobj
        % Sort individuals by objective j
        [~, sorted_idx] = sort(costs(j, :));
        % Boundary points get infinite distance
        dist(sorted_idx(1), j) = inf;
        dist(sorted_idx(end), j) = inf;
        
        % For interior points, distance = normalized difference to neighbors
        for i = 2:n-1
            dist(sorted_idx(i), j) = abs(costs(j, sorted_idx(i+1)) - costs(j, sorted_idx(i-1))) ...
                                     / (max(costs(j, :)) - min(costs(j, :)) + eps);
        end
    end
    
    % Assign total crowding distance to each individual
    for i = 1:n
        pop(idx(i)).crowdingdistance = sum(dist(i, :));
    end
end

end