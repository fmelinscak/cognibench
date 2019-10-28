function [ Prediction ] = CPC18_BEASTsd_pred( Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr )
% Prediction of BEAST.sd model for one problem
%{
This function gets as input 12 parameters which define a problem in CPC18
and outputs BEAST.sd prediction for the problem for five
blocks of five trials each (the first is without and the others are with
feedback
%}
Prediction = zeros(1,5);

% get both options' distributions
DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA);
DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB);
% get the probabilities that each option gives greater value than the other
probsBetter = get_pBetter(DistA,DistB,1,100);
% run model simulation nSims times
nSims = 10;
for sim = 1:nSims
    simPred = CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter);
    Prediction = Prediction + (1/nSims)*simPred;
end

end

