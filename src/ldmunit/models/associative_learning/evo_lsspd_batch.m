function results = evo_lsspd_batch(csInput, usInput, params)
%evo_lsspd_batch is the LSSPD batch evolution function.
%
%   evo_lsspd_batch is the LSSPD evolution function that is meant to
%   be applied in batch, i.e. for all trials at once. The LSSPD model is a
%   hybrid between Rescorla-Wagner and Pearce-Hall models. The update for
%   weights is according to RW, while the evolution of the associability is
%   according to PH. The model is named after author initals and it is described in:
%       Li, J., Schiller, D., Schoenbaum, G., Phelps, E. A., & Daw, N. D. (2011). 
%           "Differential roles of human striatum and amygdala in associative learning."
%           Nature neuroscience, 14(10), 1250-1252.
%           https://doi.org/10.1038/nn.2904
%           (supplemental material)
%
% Usage: results = evo_lsspd_batch(csInput, usInput, params)
%
% Args:
%   csInput [nTrials x nCues] : CS indicator
%   usInput [nTrials x 1] : US indicator
%   params : structure containing parameters
%       .wInit [nCues x 1] : initial associative weights
%    or .wInit : common value of initial associative weights
%       .alphaInit [nCues x 1, in [0,1]] : initial associability
%    or .alphaInit [in [0,1]]: common value of initial associability
%       .eta [in [0,1]] : proportion of prediction error signal in the updated
%           associability
%       .kappa [in [0,1]]: fixed learning rate for cue weights
% Returns:
%   results : structure wih the following fields:
%       .w [(nTrials+1) x nCues] : CS weights (together with inital ones)
%       .alpha [(nTrials+1) x nCues] : CS weights (together with inital ones)
%       .rhat [nTrials x 1] : reward prediction
%       .delta [nTrials x 1] : prediction error

% Get parameters
[nTrials, nCues] = size(csInput);

if numel(params.wInit) == 1
    wInit = repmat(params.wInit, 1, nCues); % Use same initial weight for all cues
else
    wInit = params.wInit'; % Use separate initial weights
end

if numel(params.alphaInit) == 1
    alphaInit = repmat(params.alphaInit, 1, nCues); % Use same initial associability for all cues
else
    alphaInit = params.alphaInit'; % Use separate initial associabilities
end

eta = params.eta; % Proportion of pred. error. in the updated associability value
kappa = params.kappa; % Fixed learning rate for the cue weight update

    
% Initialize results
w = nan(nTrials + 1, nCues);
w(1, :) = wInit;
alpha = nan(nTrials + 1, nCues);
alpha(1, :) = alphaInit;
rhat = nan(nTrials, 1); % Reward prediction
delta = nan(nTrials, 1); % Prediction error


% Loop over trials
for t = 1 : nTrials
    rhat(t) =  csInput(t,:) * w(t,:)'; % Predict reward
    delta(t) = usInput(t) - rhat(t); % Calculate prediction error
    w(t+1,:) =  w(t,:) + kappa*delta(t)*(alpha(t,:).*csInput(t,:)); % Update weights for active cues
    for iCue = 1 : nCues
        if csInput(t,iCue) % Cue active
            alpha(t+1,iCue) = eta*abs(delta(t)) +(1-eta)*alpha(t,iCue); % Update the associability
            alpha(t+1,iCue) = min(alpha(t+1,iCue), 1); % Enforce upper bound on alpha
        elseif ~csInput(t,iCue) % Cue inactive
            alpha(t+1,iCue) = alpha(t,iCue); % Do not update associability
        end
    end
end

% Collect results
results = struct();
results.w = w;
results.alpha = alpha;
results.rhat = rhat;
results.delta = delta;
