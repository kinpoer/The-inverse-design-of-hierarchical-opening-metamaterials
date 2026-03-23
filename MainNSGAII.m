%==========================================================================
% NSGA-II for Multi-Objective Optimization
%==========================================================================
% Description:
%   This script implements the Non-dominated Sorting Genetic Algorithm II
%   (NSGA-II) to solve a multi-objective optimization problem with 6 input
%   variables and 3 objective functions. The algorithm performs selection,
%   crossover, mutation, and elitism over multiple generations to
%   approximate the Pareto front.
%
% Author:     Ruiyi Jiang; Bo Jin
% Contact:    kinpoer@nuaa.edu.cn
% Date:       2026-03-17
% Version:    1.0
%
% Requirements:
%   - MATLAB (R2024a or later)
%   - All required functions in the 'NSGAII' folder
%   - trained_net.mat (from BP network training)
%
% Usage:
%   1. Ensure the folder 'NSGAII' is in the same directory.
%   2. Place data_HOMs.mat and trained_net.mat in the same directory.
%   3. Run the script. Pareto front updates in real-time, and final results
%      are displayed in the command window.
%==========================================================================

%% Initialization
clc; clear; close all;
addpath('NSGAII\');  % Add folder containing custom functions

%% Problem parameters
nvar = 6;                            % Number of input variables
nobj = 3;                            % Number of objective functions
npop = 50;                           % Population size
maxit = 100;                         % Maximum generations
pc = 0.85;                           % Crossover probability
nc = round(pc * npop / 2) * 2;       % Number of offspring produced by crossover (even)
mu = 0.2;                            % Mutation probability

%% Load dataset and define variable bounds
load dataset_HOMs.mat;                % Assumes 'input' is a matrix
varmin = min(input(:, 1:6));          % Lower bounds for input variables
varmax = max(input(:, 1:6));          % Upper bounds for input variables
step = abs(varmin - varmax) / 30;     % Step size for discrete variables

% Prepare parameter structure for CreateInput and Mutate functions
len = round((varmax - varmin) ./ step, 0);
var = [varmin; step; varmax; len];   % Format expected by CreateInput and Mutate

% Fixed target used in cost function (example: [78 88 0.08])
target = [78 88 0.08];

%% Initialize population structure
empty_individual.position = [];      % Input variables
empty_individual.cost = [];           % Objective values
empty_individual.rank = [];           % Non-domination rank
empty_individual.domination = [];     % Set of individuals dominated by this one
empty_individual.dominated = 0;       % Number of individuals dominating this one
empty_individual.crowdingdistance = []; % Crowding distance

pop = repmat(empty_individual, npop, 1);

%% 1. Generate initial population
for i = 1:npop
    pop(i).position = CreateInput(var);   % Create random feasible input
    pop(i).cost = Cost(pop(i).position, target); % Evaluate objectives
end

%% 2. Non-dominated sorting
[pop, F] = NonDominatedSort(pop);

%% 3. Crowding distance calculation
pop = CrowdingDistance(pop, F);

%% 4. Initialize convergence tracking arrays
avg_obj1_history = zeros(maxit, 1);
avg_obj2_history = zeros(maxit, 1);
avg_obj3_history = zeros(maxit, 1);

%% Main loop: selection, crossover, mutation, elitism
for it = 1:maxit
    % --- Crossover ---
    popc = repmat(empty_individual, nc/2, 2);  % Offspring population (pairs)
    for j = 1:nc/2
        % Tournament selection
        p1 = TourNamentSel(pop);
        p2 = TourNamentSel(pop);
        % Simulated binary crossover (SBX)
        [popc(j,1).position, popc(j,2).position] = CrossOver(p1.position, p2.position);
    end
    popc = popc(:);  % Convert to single column
    
    % --- Mutation ---
    for k = 1:nc
        popc(k).position = Mutate(popc(k).position, mu, var);
        popc(k).cost = Cost(popc(k).position, target);
    end
    
    % --- Merge parent and offspring populations ---
    combined_pop = [pop; popc];
    
    % --- Non-dominated sorting on combined population ---
    [combined_pop, F] = NonDominatedSort(combined_pop);
    
    % --- Crowding distance calculation ---
    combined_pop = CrowdingDistance(combined_pop, F);
    
    % --- Sort by rank and crowding distance ---
    combined_pop = SortPop(combined_pop);
    
    % --- Truncate to original population size (elitism) ---
    pop = combined_pop(1:npop);
    
    % --- Recompute non-dominated fronts and crowding distance for new pop ---
    [pop, F] = NonDominatedSort(pop);
    pop = CrowdingDistance(pop, F);
    pop = SortPop(pop);
    
    % --- Extract current Pareto front (rank 1 individuals) ---
    F1 = pop(F{1});
    
    % --- Record average objective values of the Pareto front ---
    if ~isempty(F1)
        costs_f1 = [F1.cost];
        avg_obj1_history(it) = mean(costs_f1(1, :));
        avg_obj2_history(it) = mean(costs_f1(2, :));
        avg_obj3_history(it) = mean(costs_f1(3, :));
    else
        % If no Pareto front (should not happen), use previous values or zero
        if it > 1
            avg_obj1_history(it) = avg_obj1_history(it-1);
            avg_obj2_history(it) = avg_obj2_history(it-1);
            avg_obj3_history(it) = avg_obj3_history(it-1);
        else
            avg_obj1_history(it) = NaN;
            avg_obj2_history(it) = NaN;
            avg_obj3_history(it) = NaN;
        end
    end
    
    % --- Display generation info ---
    fprintf('Iteration %3d: Pareto front size = %d\n', it, numel(F1));
    
    % --- Plot current Pareto front ---
    figure(9);
    PlotCosts(F1);
    drawnow;  % Force plot update
end

fprintf('\nNSGA-II optimization completed.\n');

%% Plot convergence history
figure(10);
subplot(3,1,1);
plot(1:maxit, avg_obj1_history, 'b-', 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Avg Objective 1');
title('Convergence of Pareto Front (Mean Values)');
grid on;

subplot(3,1,2);
plot(1:maxit, avg_obj2_history, 'r-', 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Avg Objective 2');
grid on;

subplot(3,1,3);
plot(1:maxit, avg_obj3_history, 'g-', 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Avg Objective 3');
grid on;

%% Select three best solutions from final Pareto front
% Define the final Pareto front (rank 1 individuals)
final_pareto = pop(F{1});

if isempty(final_pareto)
    error('No Pareto front found!');
end

% Extract cost vectors
costs_pareto = [final_pareto.cost];  % 3 x n matrix

% Compute a scalar score: sum of objectives (since they are all weighted errors)
scores = sum(costs_pareto, 1);

% Sort solutions by score (ascending)
[~, sorted_idx] = sort(scores);

% Number of solutions to select (capped by available size)
n_select = min(10, numel(final_pareto));
best_idx = sorted_idx(1:n_select);
best_solutions = final_pareto(best_idx);

%% Display the selected best solutions
fprintf('\n========================================\n');
fprintf('Top %d Solutions (by sum of objectives)\n', n_select);
fprintf('========================================\n');
for i = 1:n_select
    fprintf('\nSolution %d:\n', i);
    fprintf('  Input variables: ');
    fprintf('%8.4f ', best_solutions(i).position);
    fprintf('\n');
    fprintf('  Objectives (weighted errors): ');
    fprintf('%8.4f ', best_solutions(i).cost);
    fprintf('\n');
    fprintf('  Sum of objectives = %.4f\n', scores(best_idx(i)));
end

%% Save the best solutions to a file
save('best_solutions.mat', 'best_solutions');
fprintf('\nBest solutions saved to best_solutions.mat\n');