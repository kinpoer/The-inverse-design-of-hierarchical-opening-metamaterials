function [pop, F] = NonDominatedSort(pop)
% Perform non-dominated sorting on the population.
% Input:
%   pop - population array (struct with fields .cost, .domination, .dominated, .rank)
% Output:
%   pop - updated population with .rank, .domination, .dominated fields
%   F   - cell array where F{i} contains indices of individuals in front i
%
% Implements the fast non-dominated sort as described in NSGA-II.

npop = numel(pop);

% Initialize domination lists and counters
for i = 1:npop
    pop(i).domination = [];   % indices of individuals dominated by i
    pop(i).dominated = 0;     % number of individuals dominating i
end

% Compute pairwise domination relationships
F{1} = [];
for i = 1:npop
    for j = i+1:npop
        if Dominate(pop(i), pop(j))
            pop(i).domination = [pop(i).domination, j];
            pop(j).dominated = pop(j).dominated + 1;
        elseif Dominate(pop(j), pop(i))
            pop(j).domination = [pop(j).domination, i];
            pop(i).dominated = pop(i).dominated + 1;
        end
    end
    % Individuals with zero dominated count belong to the first front
    if pop(i).dominated == 0
        pop(i).rank = 1;
        F{1} = [F{1}, i];
    end
end

% Build subsequent fronts
k = 1;
while true
    Q = [];  % indices for next front
    for i = F{k}
        p = pop(i);
        for j = p.domination
            q = pop(j);
            q.dominated = q.dominated - 1;
            if q.dominated == 0
                q.rank = k + 1;
                Q = [Q, j];
            end
            pop(j) = q;
        end
    end
    if isempty(Q)
        break;
    else
        k = k + 1;
        F{k} = Q;
    end
end

end