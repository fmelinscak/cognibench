function [ predictions ] = CPC18_BEASTsd_pred( stimuli)
    % Prediction of BEAST.sd model for one problem
    %{
    This function gets as input 12 parameters which define a problem in CPC18
    and outputs BEAST.sd prediction for the problem for five
    blocks of five trials each (the first is without and the others are with
    feedback
    %}
    predictions = [];
    for i = 1:size(stimuli, 1)
        predictions(end + 1, :) = pred_one(stimuli(i, :));
    end
end

function prediction = pred_one(stimulus)
    Ha = stimulus(1);
    pHa = stimulus(2);
    La = stimulus(3);
    LotShapeA = stimulus(4);
    LotNumA = stimulus(5);
    Hb = stimulus(6);
    pHb = stimulus(7);
    Lb = stimulus(8);
    LotShapeB = stimulus(9);
    LotNumB = stimulus(10);
    Amb = stimulus(11);
    Corr = stimulus(12);
    prediction = zeros(1,5);
    % get both options' distributions
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA);
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB);
    % get the probabilities that each option gives greater value than the other
    probsBetter = get_pBetter(DistA,DistB,1,100);
    % run model simulation nSims times
    nSims = 10;
    for sim = 1:nSims
        simPred = CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter);
        prediction = prediction + (1/nSims)*simPred;
    end
end
