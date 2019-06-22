function results = evo_krw_batch(csInput, usInput, params)
%evo_krw_batch is the Kalman Rescorla-Wagner batch evolution function.
%
%   evo_krw_batch is the Kalman Rescorla-Wagner evolution function that is meant to
%   be applied in batch, i.e. for all trials at once. 
%   Model assumptions:
%   1) States (weights) evolve like a random walk (without deterministic dynamics or
%   inputs), with weights diffusing independently and with same variance
%   2) Observations (rewards) are noisy linear combinations of active stimuli
%   The model is described in:
%       Gershman, S.J. (2015).
%           "A unifying probabilistic view of associative learning."
%           PLoS computational biology, 11(11), p.e1004567.
%           https://doi.org/10.1371/journal.pcbi.1005829 
%
% Usage:
%   results = evo_krw_batch(csInput, usInput, params)
%
% Args:
%   csInput [nTrials x nCues] : CS indicator
%   usInput [nTrials x 1] : US indicator
%   params : structure containing parameters
%       .w [nCues x 1] : initial associative weights
%       .logTauSq : log-variance of state diffusion (common value for all
%           weights)
%       .logSigmaRSq : log-variance of the observation (reward) noise
%       .C [nCues x nCues]: CS weight covariance matrix
%       
% Returns:
%   results : structure wih the following fields:
%       .w [nCues x 1] : CS weights (together with inital ones)
%       .C [nCues x nCues] : CS weight covariance matrix
%           (together with initial one)

%% Get parameters
nCues = numel(csInput)

tauSq = exp(params.logTauSq); % State diffusion variance
Q = tauSq*eye(nCues); % Transition noise variance (transformed to positive reals); constant over time
sigmaRSq = exp(params.logSigmaRSq); % Observation noise variance
    
%% Initialize results
C_curr = C;

h_curr = csInput; % Get current CS features, which activate weights in reward prediction
w_curr = transpose(w); % Get current weights as a column vector
C_curr = C; % Get current weight covariance matrix
us_curr = usInput; % Get current US value
    
% Kalman prediction step
w_pred = w_curr; % No mean-shift for the weight distribution evolution (only stochastic evolution)
C_pred = C_curr + Q; % Update covariance
    
% Compute prediction error
rhat =  h_curr*w_pred; % Predict reward using predicted 
delta = us_curr - rhat; % Calculate prediction error

% Kalman update step
K = (C_pred*transpose(h_curr)) / (h_curr*C_pred*transpose(h_curr) + sigmaRSq);  % Kalman gain (weight-specific learning rates)
w_updt = w_pred + K*delta; % Mean updated with prediction error
C_updt = C_pred - K*h_curr*C_pred; % Covariance updated

%% Collect results
results = struct();
results.w = w_updt;
results.C = C_updt;


