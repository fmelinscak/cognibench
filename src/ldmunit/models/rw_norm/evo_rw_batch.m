function results = evo_rw_batch(csInput, usInput, params)
%evo_rw_batch is the Rescorla-Wagner batch evolution function.
%   evo_rw_batch is the Rescorla-Wagner evolution function that is meant to
%   be applied in batch, i.e. for all trials at once.
%
% Usage: results = evo_rw_batch(csInput, usInput, params)
%
% Args:
%   csInput [nTrials x nCues] : CS indicator
%   usInput [nTrials x 1] : US indicator
%   params: structure containing parameters of the model
%       .alpha [in [0,1]]: learning rate
%       .wInit [nCues x 1] : initial associative weights
%    or .wInit : common value of initial associative weights
% Returns:
%   results : structure wih the following fields:
%       .w [(nTrials+1) x nCues] : CS weights (together with inital ones)
%       .rhat [nTrials x 1] : reward prediction
%       .delta [nTrials x 1] : prediction error

% Get parameters
[nTrials, nCues] = size(csInput);
alpha = params.alpha; % Learning rate
if numel(params.wInit) == 1
    wInit = repmat(params.wInit, 1, nCues); % Use same initial weight for all cues
else
    wInit = params.wInit'; % Use separate initial weights
end
    
% Initialize results
w = nan(nTrials + 1, nCues);
w(1, :) = wInit;
rhat = nan(nTrials, 1); % Reward prediction
delta = nan(nTrials, 1); % Prediction error


% Loop over trials
for t = 1 : nTrials
    rhat(t) =  csInput(t,:) * w(t,:)'; % Predict reward
    delta(t) = usInput(t) - rhat(t); % Calculate prediction error
    w(t+1, :) =  w(t,:) + alpha * delta(t) * csInput(t,:); % Update weights for active cues
end

% Collect results
results = struct();
results.w = w;
results.rhat = rhat;
results.delta = delta;
