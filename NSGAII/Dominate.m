function z = Dominate(p, q)
% Check if individual p dominates individual q.
% Input:
%   p, q - individuals with field .cost (objective vector)
% Output:
%   z    - logical true if p dominates q, false otherwise
%
% Domination: p dominates q iff all objectives of p are <= those of q
% and at least one objective is strictly less.

cost_p = p.cost;
cost_q = q.cost;

z = all(cost_p <= cost_q) && any(cost_p < cost_q);

end