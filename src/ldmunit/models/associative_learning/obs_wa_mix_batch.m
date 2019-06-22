function results = obs_wa_mix_batch(results_evo, csInput, ~, params)
%obs_wa_mix_batch is observation function for affine transformed
%mixture of assoc. weights and associabilities
%   obs_wa_mix_batch is the observation function that generates CRs using
%   an affine transformation of the mixture of assoc. weights and 
%   associabilities for active CSs.
%
% Usage:
%   results = obs_wa_mix_batch(results_evo, csInput, ~, params)
%
% Args:
%   results_evo : structure containing the batch results of evolution
%       function
%   csInput [nTrials x nCues] : CS indicator
%   params : structure containing parameters
%       .intercept : intercept of the output mapping
%       .slope [nCues x 1] : slopes corresponding to associative weights
%    or .slope : common value of slopes corresponding to associative weights
%       .mixCoef : proportion of the weights signal in the mixture of weight and
%           associability signals
% Returns:
%   results : structure wih the following fields:
%       .crPred : predicted conditioned responses

% Get parameters
[nTrials, nCues] = size(csInput);
intercept = params.intercept; 
if numel(params.slope) == 1
    slope = repmat(params.slope, nCues, 1); % Use same slopes for all cues
else
    slope = params.slope; % Use separate slopes
end
mixCoef = params.mixCoef;
    
% Initialize results
crPred = nan(nTrials, 1);


% Loop over trials
for t = 1 : nTrials
    crPred(t) = intercept + (csInput(t,:) .* (mixCoef*results_evo.w(t,:) + (1-mixCoef)*results_evo.alpha(t,:))) * slope; % Only the active CSs contribute to CR
end

% Collect results
results = struct();
results.crPred = crPred;


end

