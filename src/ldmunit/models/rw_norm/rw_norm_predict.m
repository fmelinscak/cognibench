function [mu_pred, sd_pred] = rw_norm_predict(X, r, alpha, sigma, b0, b1, w0)

% Initialize weights and predictions
[n_trials, n_features] = size(X);
w = nan(n_trials + 1, n_features); % w(i, :) stores weights **before** 
                                   % i-th update
w(1, :) = repmat(w0, 1, n_features); % Assumes the same initial weight for
                                     % all features

mu_pred = nan(n_trials, 1);
sd_pred = repmat(sigma, n_trials, 1); % SD is always the same to the given 
                                      % parameter

% Run RW model
for i = 1 : n_trials
    % Get current weights and current cues
    w_curr = w(i, :);
    x_curr = X(i, :);
    
    % Generate outcome prediction
    rhat = x_curr * w_curr';
    
    % Predict response
    mu_pred(i) = b0 + b1*rhat;
    
    % Calculate prediction error based on observed outcome
    pred_err = r(i) - rhat;
    
    % Update weights of active cues
    w(i+1,:) = w_curr + alpha*pred_err*x_curr;
    
end