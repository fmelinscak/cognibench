function [ simPred ] = CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter)
% Simualting one virtual agent of the type BEAST.sd for one problem defined by the input
 %   Input: Payoff distributions of Option A and Option B respectively, each as a
 %   matrix of outcomes and their respective probabilities; whether ambiguity and correlation
 %   between the outcomes exists, and the probabilites that one option provides a gretaer
 %   payoff than the other.
 %   Output: the mean choice rate of option B for the current virtual agent, in blocks of 5

%% parameters
SIGMA = 13;
KAPA = 3;
BETA = 1.4;
GAMA = 1;
PSI = 0.25;
THETA = 0.7;
SIGMA_COMP = 35;
WAMB = 0.25;
%% other constants
nTrials = 25;
firstFeedback = 6;
nBlocks = 5;
%% preallocation
pBias = NaN(nTrials - firstFeedback,1);
ObsPay = NaN(nTrials - firstFeedback,2); % observed outcomes in A (col1) and B (col2)
Decision = NaN(nTrials,1);
simPred = NaN(1,nBlocks);
%% Useful variables
nA = size(DistA,1); % num outcomes in A
nB = size(DistB,1); % num outcomes in B
if Amb == 1
    ambiguous = true;
else
    ambiguous = false;
end
%% Check for problem's complexity
if (max([nA,nB]) > 2) && (min([nA,nB]) > 1)
    SIG = SIGMA_COMP;
else
    SIG = SIGMA;
end
%% draw personal traits
sigma = SIG*rand(1);
kapa = randi(KAPA);
beta = BETA*rand(1);
gama = GAMA*rand(1);
psi = PSI*rand(1);
theta = THETA*rand(1);
wamb = WAMB*rand(1);
%% more useful variables
nfeed = 0; % "t"; number of outcomes with feedback so far
pBias(nfeed+1) = beta/(beta+1+nfeed^theta);
MinA = DistA(1,1);
MinB = DistB(1,1);
MaxOutcome = max(DistA(nA,1),DistB(nB,1));
SignMax = sign(MaxOutcome);
if MinA == MinB
    RatioMin = 1;
elseif sign(MinA) == sign(MinB)
    RatioMin = min(abs(MinA),abs(MinB))/max(abs(MinA),abs(MinB));
else
    RatioMin = 0;
end
nAwin = 0; % number of times Option A's observed payoff was at least as high as B's
nBwin = 0; % number of times Option B's observed payoff was at least as high as A's
sumPayB = 0; % sum of payoffs in Option B (used if B is ambiguous)
Range = MaxOutcome - min(MinA, MinB);

%% Best estimate (0)
UEVa = (DistA(:,1)')*((1/nA)*ones(nA,1)); % EV of A had all its payoffs been equally likely
UEVb = (DistB(:,1)')*((1/nB)*ones(nB,1)); % EV of B had all its payoffs been equally likely
BEVa = (DistA(:,1)')*DistA(:,2);
if ambiguous
    BEVb = (1-psi)*mean([UEVb BEVa]) + psi*MinB;
    pEstB = zeros(nB,1); % estimation of probabilties in Amb
    t_SPminb = (BEVb -mean(DistB(2:nB,1)))/(MinB-mean(DistB(2:nB,1)));
    if t_SPminb < 0
        pEstB(1) = 0;
    elseif t_SPminb > 1
        pEstB(1) = 1;
    else
        pEstB(1) = t_SPminb;
    end
    pEstB(2:nB) = (1-pEstB(1))/(nB-1);
else
    pEstB = DistB(:,2);
    BEVb = (DistB(:,1)')*pEstB;
end
%% compute subjective dominance for this problem
subjDom = 0;
if ~ambiguous
    pAbetter = probsBetter(1);
    pBbetter = probsBetter(2);
    if (BEVa > BEVb) && (UEVa >= UEVb)
        subjDom = 1-pBbetter;
    elseif (BEVa < BEVb) && (UEVa <= UEVb)
        subjDom = 1-pAbetter;
    end
end
if (MinA > DistB(nB,1)) || (MinB > DistA(nA,1))
    subjDom = 1;
end
% correct error rate as per subjective dominance component
sigma = sigma*(1-subjDom);
sigmat = sigma;

%% simulation of the 25 decisions
for trial = 1:nTrials
    STa = 0;
    STb = 0;
    % mental simulations
    for s = 1:kapa
        rndNum = rand(2,1);
        if rndNum(1) > pBias(nfeed+1) % Unbiased technique
            if nfeed == 0
                outcomeA = distSample(DistA(:,1),DistA(:,2),rndNum(2));
                outcomeB = distSample(DistB(:,1),pEstB,rndNum(2));
            else
                uniprobs = (1/nfeed)*ones(nfeed,1);
                outcomeA = distSample(ObsPay(1:nfeed,1),uniprobs,rndNum(2));
                outcomeB = distSample(ObsPay(1:nfeed,2),uniprobs,rndNum(2));
            end
        elseif rndNum(1) > (2/3)*pBias(nfeed+1) %uniform
            outcomeA = distSample(DistA(:,1),(1/nA)*ones(nA,1),rndNum(2));
            outcomeB = distSample(DistB(:,1),(1/nB)*ones(nB,1),rndNum(2));
        elseif rndNum(1) > (1/3)*pBias(nfeed+1) %contingent pessimism
            if SignMax > 0 && RatioMin < gama
                outcomeA = MinA;
                outcomeB = MinB;
            else
                outcomeA = distSample(DistA(:,1),(1/nA)*ones(nA,1),rndNum(2));
                outcomeB = distSample(DistB(:,1),(1/nB)*ones(nB,1),rndNum(2));
            end
        else % Sign
            if nfeed == 0
                outcomeA = Range * distSample(sign(DistA(:,1)),DistA(:,2),rndNum(2));
                outcomeB = Range * distSample(sign(DistB(:,1)),pEstB,rndNum(2));
            else
                uniprobs = (1/nfeed)*ones(nfeed,1);
                outcomeA = Range * distSample(sign(ObsPay(1:nfeed,1)),uniprobs,rndNum(2));
                outcomeB = Range * distSample(sign(ObsPay(1:nfeed,2)),uniprobs,rndNum(2));
            end
        end
        STa = STa + outcomeA;
        STb = STb + outcomeB;
    end
    STa = STa/kapa;
    STb = STb/kapa;

    % error term
    error = sigmat*randn(1); % positive values contribute to attraction to A

    % decision
    Decision(trial) = (BEVa - BEVb) + (STa - STb) + error < 0;
    if (BEVa - BEVb) + (STa - STb) + error == 0
        Decision(trial) = randi(2) - 1;
    end

    % handle feedback if necessary
    if trial >= firstFeedback
        nfeed = nfeed +1;
        pBias(nfeed+1) = beta/(beta+1+nfeed^theta);
        rndNumObs = rand(1);
        ObsPay(nfeed,1) = distSample(DistA(:,1),DistA(:,2),rndNumObs); % draw outcome from A
        if Corr == 1
            ObsPay(nfeed,2) = distSample(DistB(:,1),DistB(:,2),rndNumObs);
        elseif Corr == -1
            ObsPay(nfeed,2) = distSample(DistB(:,1),DistB(:,2),1-rndNumObs);
        else
            ObsPay(nfeed,2) = distSample(DistB(:,1),DistB(:,2),rand(1)); % draw outcome from B
        end
        % update number of A and B wins
        nAwin = nAwin + (ObsPay(nfeed,1) >= ObsPay(nfeed,2));
        nBwin = nBwin + (ObsPay(nfeed,2) >= ObsPay(nfeed,1));
        sumPayB = sumPayB + ObsPay(nfeed,2);
        if ambiguous
            BEVb = (1-wamb)*BEVb + wamb*ObsPay(nfeed,2); % update best estimate of B's EV
            avgPayB = sumPayB/nfeed;
            % update size of error in ambiguous problems
            if (subjDom ~= 1)
              if ((BEVa > avgPayB) && (UEVa >= UEVb))
                sigmat = sigma * (1-nAwin/nfeed);
              elseif ((BEVa < avgPayB) && (UEVa <= UEVb))
                sigmat = sigma * (1-nBwin/nfeed);
              end
            end
        end
    end
end

% compute B-rates for this simulation
blockSize = nTrials/nBlocks;
for b = 1:nBlocks
    simPred(b) = mean(Decision(((b-1)*blockSize+1):(b*blockSize)));
end

end

