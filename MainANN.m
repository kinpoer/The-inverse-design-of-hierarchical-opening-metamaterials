%==========================================================================
% BP Neural Network for Multi-Output Regression
%==========================================================================
% Description:
%   This script trains a Backpropagation (BP) neural network with multiple
%   hidden layers to predict performance outputs based on geometric inputs.
%   The dataset is loaded from DataSetOrigin.mat, containing 'input' and
%   'output' variables. The workflow includes data splitting, normalization,
%   network training, prediction, and comprehensive evaluation (RMSE, R²,
%   MAE, MBE, MAPE) with visualizations.
%
% Author:     Ruiyi Jiang; Bo Jin
% Contact:    kinpoer@nuaa.edu.cn
% Date:       2026-03-17
% Version:    1.0
%
% Requirements:
%   - MATLAB R2024a or later
%   - Neural Network Toolbox
%
% Usage:
%   1. Place dataset_HOMs.mat in the same folder as this script.
%   2. Run the script. Results will be displayed in figures and the command window.
%==========================================================================

%% Initialization
clear; close all; clc;
warning off;  % Suppress warnings

%% Load dataset
load dataset_HOMs.mat;
data = [input, output];

%% Split ratios (must sum to 1)
train_ratio = 0.7;      % proportion for training
val_ratio   = 0.15;     % proportion for validation
test_ratio  = 0.15;     % proportion for testing
if abs(train_ratio + val_ratio + test_ratio - 1) > 1e-6
    error('Split ratios must sum to 1.');
end

%% Shuffle the dataset
num_samples = size(data, 1);
data = data(randperm(num_samples), :);

% Calculate split indices
num_train = round(train_ratio * num_samples);
num_val   = round(val_ratio   * num_samples);
num_test  = num_samples - num_train - num_val;

num_outputs = 4;                         % Number of output variables
num_features = size(data, 2) - num_outputs;

% Split data (transpose for convenience)
X_train = data(1:num_train, 1:num_features)';
Y_train = data(1:num_train, num_features+1:end)';
num_train = size(X_train, 2);

X_val   = data(num_train+1:num_train+num_val, 1:num_features)';
Y_val   = data(num_train+1:num_train+num_val, num_features+1:end)';
num_val = size(X_val, 2);

X_test  = data(num_train+num_val+1:end, 1:num_features)';
Y_test  = data(num_train+num_val+1:end, num_features+1:end)';
num_test = size(X_test, 2);

%% Normalize data to [0, 1] using training set parameters
[~, ps_input] = mapminmax([X_train, X_val, X_test], 0, 1);
X_norm_train = mapminmax('apply', X_train, ps_input);
X_norm_val   = mapminmax('apply', X_val,   ps_input);
X_norm_test  = mapminmax('apply', X_test,  ps_input);

[~, ps_output] = mapminmax([Y_train, Y_val, Y_test], 0, 1);
Y_norm_train = mapminmax('apply', Y_train, ps_output);
Y_norm_val   = mapminmax('apply', Y_val,   ps_output);
Y_norm_test  = mapminmax('apply', Y_test,  ps_output);

%% Create BP neural network
hidden_layers = [50, 50, 50, 50];
transfer_fcns = {'tansig', 'tansig', 'tansig', 'tansig', 'purelin'};
net = newff(X_norm_train, Y_norm_train, hidden_layers, transfer_fcns, 'trainlm');

%% Set training parameters
net.trainParam.epochs = 1000;      % Maximum epochs
net.trainParam.goal = 1e-4;        % Target error
net.trainParam.lr = 0.01;          % Learning rate
net.trainFcn = 'trainlm';           % Training function (Levenberg-Marquardt)

%% Train the network (only on training set)
net = train(net, X_norm_train, Y_norm_train);
% view(net);  % Uncomment to see network structure

%% Predictions
Y_pred_train_norm = sim(net, X_norm_train);
Y_pred_val_norm   = sim(net, X_norm_val);
Y_pred_test_norm  = sim(net, X_norm_test);

%% Denormalize predictions
Y_pred_train = mapminmax('reverse', Y_pred_train_norm, ps_output);
Y_pred_val   = mapminmax('reverse', Y_pred_val_norm,   ps_output);
Y_pred_test  = mapminmax('reverse', Y_pred_test_norm,  ps_output);

%% Save the trained network and parameters
save('trained_net.mat', 'net', 'ps_input', 'ps_output');

%% Evaluation and visualization for each output
for k = 1:num_outputs
    % ----- Training set metrics -----
    rmse_train = sqrt(mean((Y_pred_train(k,:) - Y_train(k,:)).^2));
    R2_train   = 1 - sum((Y_train(k,:) - Y_pred_train(k,:)).^2) / ...
                     sum((Y_train(k,:) - mean(Y_train(k,:))).^2);
    mae_train  = mean(abs(Y_pred_train(k,:) - Y_train(k,:)));
    mbe_train  = mean(Y_pred_train(k,:) - Y_train(k,:));
    mape_train = mean(abs((Y_train(k,:) - Y_pred_train(k,:)) ./ Y_train(k,:))) * 100;

    % ----- Validation set metrics -----
    rmse_val   = sqrt(mean((Y_pred_val(k,:)   - Y_val(k,:)).^2));
    R2_val     = 1 - sum((Y_val(k,:)   - Y_pred_val(k,:)).^2) / ...
                     sum((Y_val(k,:)   - mean(Y_val(k,:))).^2);
    mae_val    = mean(abs(Y_pred_val(k,:)   - Y_val(k,:)));
    mbe_val    = mean(Y_pred_val(k,:)   - Y_val(k,:));
    mape_val   = mean(abs((Y_val(k,:)   - Y_pred_val(k,:)) ./ Y_val(k,:))) * 100;

    % ----- Test set metrics -----
    rmse_test  = sqrt(mean((Y_pred_test(k,:)  - Y_test(k,:)).^2));
    R2_test    = 1 - sum((Y_test(k,:)  - Y_pred_test(k,:)).^2) / ...
                     sum((Y_test(k,:)  - mean(Y_test(k,:))).^2);
    mae_test   = mean(abs(Y_pred_test(k,:)  - Y_test(k,:)));
    mbe_test   = mean(Y_pred_test(k,:)  - Y_test(k,:));
    mape_test  = mean(abs((Y_test(k,:)  - Y_pred_test(k,:)) ./ Y_test(k,:))) * 100;

    % Print results to console
    fprintf('\n**************************\n');
    fprintf('Output %d\n', k);
    fprintf('**************************\n');
    fprintf('Training   - RMSE: %.4f, R²: %.4f, MAE: %.4f, MBE: %.4f, MAPE: %.2f%%\n', ...
            rmse_train, R2_train, mae_train, mbe_train, mape_train);
    fprintf('Validation - RMSE: %.4f, R²: %.4f, MAE: %.4f, MBE: %.4f, MAPE: %.2f%%\n', ...
            rmse_val,   R2_val,   mae_val,   mbe_val,   mape_val);
    fprintf('Test       - RMSE: %.4f, R²: %.4f, MAE: %.4f, MBE: %.4f, MAPE: %.2f%%\n', ...
            rmse_test,  R2_test,  mae_test,  mbe_test,  mape_test);

    % ----- Visualizations -----
    % 1. Prediction comparison (three subplots)
    figure;
    subplot(3,1,1);
    plot(1:num_train, Y_train(k,:), 'r-*', 1:num_train, Y_pred_train(k,:), 'b-o', 'LineWidth',1);
    legend('True','Predicted','Location','best');
    xlabel('Sample index'); ylabel(['Output ' num2str(k)]);
    title(sprintf('Training Set - Output %d (RMSE = %.4f)', k, rmse_train));
    grid on; xlim([1 num_train]);

    subplot(3,1,2);
    plot(1:num_val, Y_val(k,:), 'r-*', 1:num_val, Y_pred_val(k,:), 'b-o', 'LineWidth',1);
    legend('True','Predicted','Location','best');
    xlabel('Sample index'); ylabel(['Output ' num2str(k)]);
    title(sprintf('Validation Set - Output %d (RMSE = %.4f)', k, rmse_val));
    grid on; xlim([1 num_val]);

    subplot(3,1,3);
    plot(1:num_test, Y_test(k,:), 'r-*', 1:num_test, Y_pred_test(k,:), 'b-o', 'LineWidth',1);
    legend('True','Predicted','Location','best');
    xlabel('Sample index'); ylabel(['Output ' num2str(k)]);
    title(sprintf('Test Set - Output %d (RMSE = %.4f)', k, rmse_test));
    grid on; xlim([1 num_test]);

    % 2. Error histograms (three separate figures)
    figure; ploterrhist(Y_train(k,:) - Y_pred_train(k,:), sprintf('Training Set - Output %d', k));
    figure; ploterrhist(Y_val(k,:)   - Y_pred_val(k,:),   sprintf('Validation Set - Output %d', k));
    figure; ploterrhist(Y_test(k,:)  - Y_pred_test(k,:),  sprintf('Test Set - Output %d', k));

    % 3. Linear fit plots – all three sets in one figure
    figure;
    hold on;
    % Training
    plot(Y_train(k,:), Y_pred_train(k,:), 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    p_train = polyfit(Y_train(k,:), Y_pred_train(k,:), 1);
    x_fit = linspace(min(Y_train(k,:)), max(Y_train(k,:)), 100);
    y_fit = polyval(p_train, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 1);
    % Validation
    plot(Y_val(k,:), Y_pred_val(k,:), 'g', 'filled', 'MarkerFaceAlpha', 0.3);
    p_val = polyfit(Y_val(k,:), Y_pred_val(k,:), 1);
    x_fit = linspace(min(Y_val(k,:)), max(Y_val(k,:)), 100);
    y_fit = polyval(p_val, x_fit);
    plot(x_fit, y_fit, 'g-', 'LineWidth', 1);
    % Test
    plot(Y_test(k,:), Y_pred_test(k,:), 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    p_test = polyfit(Y_test(k,:), Y_pred_test(k,:), 1);
    x_fit = linspace(min(Y_test(k,:)), max(Y_test(k,:)), 100);
    y_fit = polyval(p_test, x_fit);
    plot(x_fit, y_fit, 'b-', 'LineWidth', 1);
    xlabel('True value'); ylabel('Predicted value');
    legend('Training', 'Training fit', ...
           'Validation', 'Validation fit', ...
           'Test', 'Test fit', 'Location', 'best');
    title(sprintf('Linear Fit Comparison - Output %d', k));
    axis equal tight; grid on;
end

fprintf('\nAll done. Trained network saved as trained_net.mat\n');