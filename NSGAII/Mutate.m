function y = Mutate(x, mu, var)
% Mutation operator for NSGA-II.
% Input:
%   x   - 1xn individual to mutate
%   mu  - mutation probability per gene
%   var - variable definition matrix (see CreateInput)
% Output:
%   y   - mutated individual
%
% Each gene is independently mutated with probability mu.
% If mutated, the gene is replaced by the corresponding value from a
% newly generated random feasible vector.

n = numel(x);
% Determine which genes to mutate
r = rand(1, n);
mutate_idx = find(r <= mu);

if ~isempty(mutate_idx)
    % Generate a completely new random feasible vector
    new_individual = CreateInput(var);
    % Replace mutated positions
    x(mutate_idx) = new_individual(mutate_idx);
end

y = x;

end