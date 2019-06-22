function results = obs_affine_batch(results_evo, csInput, ~, params)
%obs_affine_batch is observation function for affine transformed weights
%   obs_affine_batch is the observation function that generates CRs using
%   an affine transformation of the active weights
%
% Usage:
%   results = obs_affine_batch(results_evo, csInput, ~, params)
%
% Args:
%   results_evo : structure containing the batch results of evolution
%       function
%   csInput [nTrials x nCues] : CS indicator
%   params : structure containing parameters
%       .intercept : intercept of the output mapping
%       .slope [nCues x 1] : slopes corresponding to associative weights
%    or .slope : common value of slopes corresponding to associative weights
%
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
    
% Initialize results
crPred = nan(nTrials, 1);


% Loop over trials
for t = 1 : nTrials
    crPred(t) = intercept + (csInput(t,:) .* results_evo.w(t,:)) * slope; % Only the active CSs contribute to CR
end

% Collect results
results = struct();
results.crPred = crPred;


end

