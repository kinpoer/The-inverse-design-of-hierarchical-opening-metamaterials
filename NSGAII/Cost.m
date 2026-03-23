function z = Cost(x, target)
% Cost function for NSGA-II individuals.
% Input:
%   x      - 1xn vector of input variables (design variables)
%   target - 1x3 vector of target objective values
% Output:
%   z      - 3x1 vector of weighted absolute errors (to be minimized)
%
% The function loads a pre-trained neural network (trained_net.mat) and
% uses it to predict the output for the given input. The prediction is
% then compared to the target values using weighted absolute errors.

% Load trained network and normalization parameters
load('trained_net.mat', 'net', 'ps_input', 'ps_output');

% Normalize input using the same parameters as during training
x_norm = mapminmax('apply', x', ps_input);  % x' converts to column vector

% Predict using the neural network
try
    y_norm = predict(net, x_norm')';% Built‑in network function
catch
    y_norm = sim(net,x_norm);% Custom network function
end

y = mapminmax('reverse', y_norm, ps_output); % Denormalized prediction (3x1)

% Target values
obj1_target = target(1);
obj2_target = target(2);
obj3_target = target(3);

% Weights for each objective (can be adjusted)
w1 = 1.0;
w2 = 1.0;
w3 = 1.0;

% Weighted absolute percentage errors
error1 = w1 * abs(y(1) - obj1_target) / obj1_target;
error2 = w2 * abs(y(2) - obj2_target) / obj2_target;
error3 = w3 * abs(y(3) - obj3_target) / obj3_target;

z = [error1; error2; error3];

end