function [mu_pred, sd_pred] = krw_norm_predict(X, r, sigma_a, sigma_r, sigma_w, sigma_w0, b0, b1, w0)

% Initialize weights and weight covariance
[n_trials, n_features] = size(X);
w = nan(n_trials + 1, n_features); % w(i, :) stores weights **before** 
                                   % i-th update
w(1, :) = repmat(w0, 1, n_features); % Assumes the same initial weight for
                                     % all features
                                     
C = nan(n_features, n_features, n_trials + 1);
C(:, :, 1) = (sigma_w0^2)*eye(n_features); % Initial weight covariance matrix

Q = (sigma_w^2)*eye(n_features); % Transition noise variance; constant over time


% Initialize predictions
mu_pred = nan(n_trials, 1);
sd_pred = repmat(sigma_a, n_trials, 1); % SD is always the same to the given 
                                        % parameter

% Run KRW model
for i = 1 : n_trials
    % Get current weights, covariance, and current cues
    w_curr = w(i, :)';
    C_curr = C(:, :, i); % Get current weight covariance matrix
    x_curr = X(i, :);
    
    % Generate outcome prediction
    w_pred = w_curr; % No mean-shift for the weight distribution evolution (only stochastic evolution)
    C_pred = C_curr + Q; % Update covariance
    rhat = x_curr * w_pred;
    
    % Predict response
    mu_pred(i) = b0 + b1*rhat;
    
    % Calculate prediction error based on observed outcome
    pred_err = r(i) - rhat;
    
    % Kalman update step
    K = (C_pred*x_curr') / (x_curr*C_pred*x_curr' + sigma_r^2);  % Kalman gain (weight-specific learning rates)
    w(i+1, :) = w_pred + K*pred_err; % Mean updated with prediction error
    C(:, :, i+1) = C_pred - K*x_curr*C_pred; % Covariance updated
    
end
