CPC18_BEASTsd_pred = function(stimuli) {
  #  This function gets as input 12 parameters which define a problem in CPC18
  #  and outputs BEAST.sd model's prediction in that problem for five blocks of
  #  five trials each (the first is without and the others are with feedback
  return(t(apply(stimuli, 1, pred_one)))
}

pred_one = function(stimulus) {
  Ha <- stimulus[[1]]
  pHa <- stimulus[[2]]
  La <- stimulus[[3]]
  LotShapeA <- stimulus[[4]]
  LotNumA <- stimulus[[5]]
  Hb <- stimulus[[6]]
  pHb <- stimulus[[7]]
  Lb <- stimulus[[8]]
  LotShapeB <- stimulus[[9]]
  LotNumB <- stimulus[[10]]
  Amb <- stimulus[[11]]
  Corr <- stimulus[[12]]
  Prediction = rep(0,5)

  # get both options' detailed distributions
  DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
  DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

  # get the probabilities that each option gives greater value than the other
  probsBetter = get_pBetter(DistA,DistB,corr=1, accuracy = 100)
  # run model simulation nSims times
  nSims = 10;
  for (sim in 1:nSims){
    simPred = CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter)
    Prediction = Prediction + (1/nSims)*simPred
  }
  return(Prediction)
}
