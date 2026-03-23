function PlotCosts(pop)
% Plot the Pareto front in 3D.
% Input:
%   pop - population array (struct with .cost field)
%
% The function extracts the objective vectors and creates a 3D scatter plot.
% Color bar indicates a fourth objective (if present), otherwise just for decoration.

costs = [pop.cost];  % 3 x n matrix

% Set figure properties
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

% Create 3D scatter plot
scatter3(costs(1, :), costs(2, :), costs(3, :), 50, 'filled', 'ro');
xlabel('Objective 1');
ylabel('Objective 2');
zlabel('Objective 3');
title('Pareto Front');

% Add color bar (optional, here labeled as y4 for consistency with original)
c = colorbar;
c.Label.String = 'y4';
grid on;

end