%==========================================================================
% BP Neural Network for Multi-Output Regression
%==========================================================================
% Description:
%   This script trains a simple neural network with multiple
%   hidden layers to predict performance outputs based on geometric inputs.
%   The dataset is loaded from dataset_HOMs.mat, containing 'input' and
%   'output' variables. The dataset is split into training, valididation, 
%   and test sets with user‑defined ratios.The workflow includes data 
%   splitting, normalization, network training, prediction, and comprehensive
%   evaluation (RMSE, R², MAE, MBE, MAPE) with visualizations. The trained 
%   model is evaliduated on all three sets using standard regression metrics.
%
% Author:     Ruiyi Jiang; Bo Jin
% Contact:    kinpoer@nuaa.edu.cn
% Date:       2026-03-17
% Version:    1.0
%
% Requirements:
%   - MATLAB R2024a or later
%   - Deep Learning Toolbox
%
% Usage:
%   1. Place dataset_HOMs.mat in the same folder as this script.
%   2. Run the script. Results will be displayed in figures and the command window.
%==========================================================================

%% Environment setup
close all;clear;clc
warning off; % Suppress warnings

%% Load dataset
load dataset_HOMs.mat; % input：openings, thickness; output：MCS, IPCS, PE, m
outdim = size(output, 2); % Automatically detect number of outputs
X = input; Y = output;
num_samples = size(X, 1);

%% Split ratios (must sum to 1)
train_ratio = 0.7; val_ratio   = 0.15; test_ratio  = 0.15;
if abs(train_ratio + val_ratio + test_ratio - 1) > 1e-6
    error('Split ratios must sum to 1.');
end

%% Shuffle the dataset
idx = randperm(num_samples);
X = X(idx, :); Y = Y(idx, :);

% Calculate split indices
num_train = round(train_ratio * num_samples); 
num_val   = round(val_ratio   * num_samples);
num_test  = num_samples - num_train - num_val;

% Split data (transpose to have features × samples for trainnet)
X_train = X(1:num_train, :)'; 
Y_train = Y(1:num_train, :)';
num_train = size(X_train, 2);

X_val   = X(num_train+1:num_train+num_val, :)';
Y_val   = Y(num_train+1:num_train+num_val, :)';
num_val = size(X_val, 2);

X_test  = X(num_train+num_val+1:end, :)';
Y_test  = Y(num_train+num_val+1:end, :)';
num_test = size(X_test, 2);

num_features = size(X_train, 1);

%% Normalize data using training set parameters (min‑max scaling to [0,1])
[~, ps_input] = mapminmax([X_train, X_val, X_test], 0, 1);
X_norm_train = mapminmax('apply', X_train, ps_input);
X_norm_val   = mapminmax('apply', X_val, ps_input);
X_norm_test = mapminmax('apply', X_test, ps_input);

[~, ps_output] = mapminmax([Y_train, Y_val, Y_test], 0, 1);
Y_norm_train = mapminmax('apply', Y_train, ps_output);
Y_norm_val   = mapminmax('apply', Y_val, ps_output);
Y_norm_test = mapminmax('apply', Y_test, ps_output);


%% Build the deep neural network
hidden_units = [50, 50, 50, 50, 50];
leaky_alpha = 0.01;

layers = featureInputLayer(num_features, 'Name', 'input');

for i = 1:numel(hidden_units)
    layers = [layers
        fullyConnectedLayer(hidden_units(i), 'Name', sprintf('fc_%d', i))
        batchNormalizationLayer('Name', sprintf('bn_%d', i))
        leakyReluLayer(leaky_alpha, 'Name', sprintf('leaky_relu_%d', i))
    ];
end
layers = [layers
    % fullyConnectedLayer(16, 'Name', 'pre_output')
    % batchNormalizationLayer('Name', 'bn_pre_output')
    % leakyReluLayer(leaky_alpha, 'Name', 'relu_pre_output')
    fullyConnectedLayer(outdim, 'Name', 'output')
];

% Visualize network structure (optional)
analyzeNetwork(layers);

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.5, ...
    'L2Regularization', 0.0001, ...
    'ValidationData', {X_norm_val', Y_norm_val'}, ...
    'ValidationFrequency', 120, ...
    'ValidationPatience',  35, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Metrics', 'rmse', ...
    'OutputFcn', @(info) saveTrainingInfo(info));

%% Train the network
net = trainnet(X_norm_train', Y_norm_train', layers, 'mse', options);

%% Predict and denormalize
Y_pred_train_norm = predict(net, X_norm_train')';
Y_pred_val_norm   = predict(net, X_norm_val')';
Y_pred_test_norm  = predict(net, X_norm_test')';

Y_pred_train = mapminmax('reverse', Y_pred_train_norm, ps_output);
Y_pred_val   = mapminmax('reverse', Y_pred_val_norm,   ps_output);
Y_pred_test  = mapminmax('reverse', Y_pred_test_norm,  ps_output);

%% Save the trained network and normalization parameters
save('trained_net.mat', 'net', 'ps_input', 'ps_output');

%% Plot training history 
plotTrainingHistory(trainingHistory)

%% Evaluate each output dimension for all three splits
for k = 1:outdim
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
    figure; ploterrhist(Y_train(k,:) - Y_pred_train(k,:), sprintf('Training Set - Output %d', k)); ylabel('Count');
    figure; ploterrhist(Y_val(k,:)   - Y_pred_val(k,:),   sprintf('Validation Set - Output %d', k)); ylabel('Count');
    figure; ploterrhist(Y_test(k,:)  - Y_pred_test(k,:),  sprintf('Test Set - Output %d', k)); ylabel('Count');

    % 3. Linear fit plots – all three sets in one figure
    figure;hold on;
    % Training
    scatter(Y_train(k,:), Y_pred_train(k,:), 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    p_train = polyfit(Y_train(k,:), Y_pred_train(k,:), 1);
    x_fit = linspace(min(Y_train(k,:)), max(Y_train(k,:)), 100);
    y_fit = polyval(p_train, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    % Validation
    scatter(Y_val(k,:), Y_pred_val(k,:), 'g', 'filled', 'MarkerFaceAlpha', 0.3);
    p_val = polyfit(Y_val(k,:), Y_pred_val(k,:), 1);
    x_fit = linspace(min(Y_val(k,:)), max(Y_val(k,:)), 100);
    y_fit = polyval(p_val, x_fit);
    plot(x_fit, y_fit, 'g-', 'LineWidth', 2);
    % Test
    scatter(Y_test(k,:), Y_pred_test(k,:), 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    p_test = polyfit(Y_test(k,:), Y_pred_test(k,:), 1);
    x_fit = linspace(min(Y_test(k,:)), max(Y_test(k,:)), 100);
    y_fit = polyval(p_test, x_fit);
    plot(x_fit, y_fit, 'b-', 'LineWidth', 2);
    xlabel('True value'); ylabel('Predicted value');
    legend('Training', 'Training fit', ...
           'Validation', 'Validation fit', ...
           'Test', 'Test fit', 'Location', 'best');
    title(sprintf('Linear Fit Comparison - Output %d', k));
    axis equal tight; box on; 
end

fprintf('\nAll done. Trained network saved as trained_net.mat\n');

%==========================================================================
% Output function to record training history
%==========================================================================
function stop = saveTrainingInfo(info)
% Save history data
    persistent history   % Persistent variable to store history

    % Initialize history structure if empty
    if isempty(history)
        history = struct(...
            'Epoch', [], ...
            'Iteration', [], ...
            'TrainingLoss', [], ...
            'TrainingAccuracy', [], ...
            'ValidationLoss', [], ...
            'ValidationAccuracy', [], ...
            'ValidationFrequency', [], ...
            'LearningRate', [], ...
            'TrainingTime', [], ...
            'TrainingRMSE', [], ...
            'ValidationRMSE', [], ...
            'TrainingMAE', [], ...
            'ValidationMAE', [], ...
            'GradientNorm', [], ...
            'IterationTime', []);
    end
    
    % Record when new training information is available
    if ~isempty(info.TrainingLoss)
        idx = length(history.Epoch) + 1;
        
        % Basic training info
        history.Epoch(idx) = info.Epoch;
        history.Iteration(idx) = info.Iteration;
        history.TrainingLoss(idx) = info.TrainingLoss;
        
        % Training accuracy (if exists)
        if isfield(info, 'TrainingAccuracy') && ~isempty(info.TrainingAccuracy)
            history.TrainingAccuracy(idx) = info.TrainingAccuracy;
        else
            history.TrainingAccuracy(idx) = NaN;
        end
        
        % Validation loss (if exists)
        if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
            history.ValidationLoss(idx) = info.ValidationLoss;
        else
            history.ValidationLoss(idx) = NaN;
        end
        
        % Validation accuracy (if exists)
        if isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
            history.ValidationAccuracy(idx) = info.ValidationAccuracy;
        else
            history.ValidationAccuracy(idx) = NaN;
        end
        
        % Learning rate (if exists)
        if isfield(info, 'LearnRate') && ~isempty(info.LearnRate)
            history.LearningRate(idx) = info.LearnRate;
        else
            history.LearningRate(idx) = NaN;
        end
        
        % Training time (if exists)
        if isfield(info, 'Time') && ~isempty(info.Time)
            history.TrainingTime(idx) = info.Time;
        else
            history.TrainingTime(idx) = NaN;
        end
        
        % Validation frequency (if exists)
        if isfield(info, 'ValidationFrequency') && ~isempty(info.ValidationFrequency)
            history.ValidationFrequency(idx) = info.ValidationFrequency;
        else
            history.ValidationFrequency(idx) = NaN;
        end
        
        % Root Mean Square Error (RMSE) for regression tasks
        if isfield(info, 'TrainingRMSE') && ~isempty(info.TrainingRMSE)
            history.TrainingRMSE(idx) = info.TrainingRMSE;
        else
            history.TrainingRMSE(idx) = NaN;
        end
        
        if isfield(info, 'ValidationRMSE') && ~isempty(info.ValidationRMSE)
            history.ValidationRMSE(idx) = info.ValidationRMSE;
        else
            history.ValidationRMSE(idx) = NaN;
        end
        
        % Mean Absolute Error (MAE) for regression tasks
        if isfield(info, 'TrainingMAE') && ~isempty(info.TrainingMAE)
            history.TrainingMAE(idx) = info.TrainingMAE;
        else
            history.TrainingMAE(idx) = NaN;
        end
        
        if isfield(info, 'ValidationMAE') && ~isempty(info.ValidationMAE)
            history.ValidationMAE(idx) = info.ValidationMAE;
        else
            history.ValidationMAE(idx) = NaN;
        end
        
        % Gradient norm (if exists)
        if isfield(info, 'GradientNorm') && ~isempty(info.GradientNorm)
            history.GradientNorm(idx) = info.GradientNorm;
        else
            history.GradientNorm(idx) = NaN;
        end
        
        % Iteration time (if exists)
        if isfield(info, 'IterationTime') && ~isempty(info.IterationTime)
            history.IterationTime(idx) = info.IterationTime;
        else
            history.IterationTime(idx) = NaN;
        end
        
        % Save to base workspace
        assignin('base', 'trainingHistory', history);
        
        % Optional: automatically save to file
        % saveTrainingHistoryToFile(history);
    end
    
    stop = false; % Continue training
end

%==========================================================================
% Function to plot training history
%==========================================================================
function plotTrainingHistory(varargin)
% Get history data
if nargin == 0
    history = evalin('base', 'trainingHistory');
    if isempty(history)
        error('Variable "trainingHistory" not found. Please run the training script first.');
    end
else
    history = varargin{1};
end

% Extract common fields
iter = history.Iteration;
trainLoss = history.TrainingLoss;
valLoss = history.ValidationLoss;
trainRMSE = history.TrainingRMSE;
valRMSE = history.ValidationRMSE;
trainMAE = history.TrainingMAE;
valMAE = history.ValidationMAE;
lr = history.LearningRate;
grad = history.GradientNorm;
iterTime = history.IterationTime;
epoch = history.Epoch;

% Remove NaNs from validation data
validValLoss = ~isnan(valLoss);
iterValLoss = iter(validValLoss);
valLossValid = valLoss(validValLoss);

validValRMSE = ~isnan(valRMSE);
iterValRMSE = iter(validValRMSE);
valRMSEValid = valRMSE(validValRMSE);

validValMAE = ~isnan(valMAE);
iterValMAE = iter(validValMAE);
valMAEValid = valMAE(validValMAE);

% Check if any validation data exists
hasVal = any(validValLoss) || any(validValRMSE) || any(validValMAE);

% Create figure for loss and RMSE
fig1 = figure('Name', 'Loss and RMSE Convergence', 'NumberTitle', 'off');
if hasVal
    subplot(2,1,1);
else
    subplot(1,1,1);
end

% Loss curve
plot(iter, trainLoss, 'b-', 'LineWidth', 1.5); hold on;
if any(validValLoss)
    plot(iterValLoss, valLossValid, 'r--', 'LineWidth', 1.5);
    legend('Training Loss', 'Validation Loss', 'Location', 'best');
else
    legend('Training Loss', 'Location', 'best');
end
xlabel('Iteration'); ylabel('Loss (MSE)'); set(gca, 'YScale', 'log');
title('Loss Convergence');
grid on;

% RMSE curve
if hasVal
    subplot(2,1,2);
    plot(iter, trainRMSE, 'b-', 'LineWidth', 1.5); hold on;
    if any(validValRMSE)
        plot(iterValRMSE, valRMSEValid, 'r--', 'LineWidth', 1.5);
        legend('Training RMSE', 'Validation RMSE', 'Location', 'best');
    else
        legend('Training RMSE', 'Location', 'best');
    end
    xlabel('Iteration'); ylabel('RMSE');  set(gca, 'YScale', 'log');
    title('RMSE Convergence');
    grid on;
end

% MAE curve (if present)
if ~all(isnan(trainMAE))
    fig2 = figure('Name', 'MAE Convergence', 'NumberTitle', 'off');
    plot(iter, trainMAE, 'b-', 'LineWidth', 1.5); hold on;
    if any(validValMAE)
        plot(iterValMAE, valMAEValid, 'r--', 'LineWidth', 1.5);
        legend('Training MAE', 'Validation MAE', 'Location', 'best');
    else
        legend('Training MAE', 'Location', 'best');
    end
    xlabel('Iteration'); ylabel('MAE');
    title('MAE Convergence');
    grid on;
end

% Learning rate curve
if ~all(isnan(lr))
    fig3 = figure('Name', 'Learning Rate Schedule', 'NumberTitle', 'off');
    plot(iter, lr, 'k-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Learning Rate');
    title('Learning Rate Schedule');
    grid on; set(gca, 'YScale', 'log');
end

% Gradient norm curve
if ~all(isnan(grad))
    fig4 = figure('Name', 'Gradient Norm', 'NumberTitle', 'off');
    plot(iter, grad, 'm-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Gradient Norm');
    title('Gradient Norm');
    grid on;
end

% Iteration time curve (optional)
if ~all(isnan(iterTime))
    fig5 = figure('Name', 'Iteration Time', 'NumberTitle', 'off');
    plot(iter, iterTime, 'g-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Time per Iteration (s)');
    title('Iteration Time');
    grid on;
end

% Epoch-wise aggregation (if epoch field is available)
if ~isempty(epoch) && all(~isnan(epoch))
    [uniqueEpochs, ~, idx] = unique(epoch);
    lastIterIdx = accumarray(idx, 1:length(epoch), [], @max);
    
    epochLoss = trainLoss(lastIterIdx);
    epochRMSE = trainRMSE(lastIterIdx);
    epochMAE  = trainMAE(lastIterIdx);
    epochValLoss = valLoss(lastIterIdx);
    epochValRMSE = valRMSE(lastIterIdx);
    epochValMAE  = valMAE(lastIterIdx);
    
    % Remove NaNs from epoch‑wise validation data
    validEpochValLoss = ~isnan(epochValLoss);
    epochValLossValid = epochValLoss(validEpochValLoss);
    epochValEpochs = uniqueEpochs(validEpochValLoss);
    
    validEpochValRMSE = ~isnan(epochValRMSE);
    epochValRMSEValid = epochValRMSE(validEpochValRMSE);
    epochValRMSEEpochs = uniqueEpochs(validEpochValRMSE);
    
    fig6 = figure('Name', 'Epoch-wise Metrics', 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(uniqueEpochs, epochLoss, 'b-o', 'LineWidth', 1.5); hold on;
    if any(validEpochValLoss)
        plot(epochValEpochs, epochValLossValid, 'r-s', 'LineWidth', 1.5);
        legend('Training Loss', 'Validation Loss', 'Location', 'best');
    else
        legend('Training Loss', 'Location', 'best');
    end
    xlabel('Epoch'); ylabel('Loss (MSE)');
    title('Epoch-wise Loss');
    grid on;
    
    subplot(2,1,2);
    plot(uniqueEpochs, epochRMSE, 'b-o', 'LineWidth', 1.5); hold on;
    if any(validEpochValRMSE)
        plot(epochValRMSEEpochs, epochValRMSEValid, 'r-s', 'LineWidth', 1.5);
        legend('Training RMSE', 'Validation RMSE', 'Location', 'best');
    else
        legend('Training RMSE', 'Location', 'best');
    end
    xlabel('Epoch'); ylabel('RMSE');
    title('Epoch-wise RMSE');
    grid on;
end

fprintf('Training history visualization completed.\n');
end