CPC18_BEASTsd_simulation = function (DistA, DistB, Amb, Corr, probsBetter){
  # Simualting one virtual agent of the type BEAST.sd for one problem defined by the input
  #   Input: Payoff distributions of Option A and Option B respectively, each as a
  #   matrix of outcomes and their respective probabilities; whether ambiguity and correlation
  #   between the outcomes exists, and the probabilites that one option provides a gretaer
  #   payoff than the other.
  #   Output: the mean choice rate of option B for the current virtual agent, in blocks of 5

  # Model free parameters
  SIGMA = 13
  KAPA = 3
  BETA = 1.4
  GAMA = 1
  PSI = 0.25
  THETA = 0.7
  SIGMA_COMP = 35
  WAMB = 0.25

  # Setting's constants used
  nTrials = 25
  firstFeedback = 6
  nBlocks = 5

  # Useful variables
  nA = nrow(DistA) # num outcomes in A
  nB = nrow(DistB) # num outcomes in B
  if (Amb == 1) {
    ambiguous = TRUE}
  else  {
    ambiguous = FALSE}

  # Initialize variables
  pBias = rep(NA, nTrials - firstFeedback+1) # probability of choosing biased simulation tool
  ObsPay = matrix(NA,nrow=nTrials - firstFeedback+1,ncol=2) # observed outcomes in A (col1) and B (col2)
  Decision = rep(NA,nTrials) #vector of decisions of the agent
  simPred = matrix(NA,nrow=1,ncol=nBlocks)

  # check for complexity of problem
  if ((max(c(nA,nB)) > 2) && (min(c(nA,nB)) > 1)){
    SIG = SIGMA_COMP}
  else {
    SIG = SIGMA
  }

  # draw personal traits
  sigma = SIG*runif(1)
  kapa = sample(KAPA,1)
  beta = BETA*runif(1)
  gama = GAMA*runif(1)
  psi = PSI*runif(1)
  theta = THETA*runif(1)
  wamb = WAMB*runif(1)

  # More useful variables
  nfeed = 0 # "t"; number of outcomes with feedback so far
  pBias[nfeed+1] = beta/(beta+1+nfeed^theta)
  MinA = DistA[1,1]
  MinB = DistB[1,1]
  MaxOutcome = max(DistA[nA,1],DistB[nB,1])
  SignMax = sign(MaxOutcome)
  #Compute "RatioMin"
  if (MinA == MinB){
    RatioMin = 1}
  else if (sign(MinA) == sign(MinB)){
    RatioMin = min(abs(MinA),abs(MinB))/max(abs(MinA),abs(MinB))}
  else {
    RatioMin = 0
  }
  nAwin = 0 # number of times Option A's observed payoff was at least as high as B's
  nBwin = 0 # number of times Option B's observed payoff was at least as high as A's
  sumPayB = 0 # sum of payoffs in Option B (used if B is abmiguous)
  Range = MaxOutcome - min(MinA, MinB)

  UEVa = DistA[,1]%*%(rep(1/nA,nA)) # EV of A had all its payoffs been equally likely
  UEVb = DistB[,1]%*%(rep(1/nB,nB)) # EV of B had all its payoffs been equally likely
  BEVa = DistA[,1]%*%DistA[,2] # Best estimate of EV of Option B
  if (ambiguous){
    BEVb = (1-psi)*(UEVb+BEVa)/2 + psi*MinB
    pEstB = rep(nB,1) # estimation of probabilties in Amb
    t_SPminb = (BEVb -mean(DistB[2:nB,1]))/(MinB-mean(DistB[2:nB,1]))
    if (t_SPminb < 0 ){
      pEstB[1] = 0}
    else if (t_SPminb > 1){
      pEstB[1] = 1}
    else{
      pEstB[1] = t_SPminb
    }
    pEstB[2:nB] = (1-pEstB[1])/(nB-1)}
  else {
    pEstB = DistB[,2]
    BEVb = DistB[,1]%*%pEstB
  }

  # compute subjective dominance for this problem
  subjDom = 0
  if (!ambiguous){
    pAbetter = probsBetter[[1]]
    pBbetter = probsBetter[[2]]
    if ((BEVa > BEVb) & (UEVa >= UEVb)){
      subjDom = 1-pBbetter}
    else if ((BEVa < BEVb) & (UEVa <= UEVb)){
      subjDom = 1-pAbetter}
  }
  if ((MinA > DistB[nB,1]) | (MinB > DistA[nA,1])){
    subjDom = 1
  }
  # correct error rate as per subjective dominance component
  sigma = sigma*(1-subjDom)
  sigmat = sigma

  # simulation of the 25 decisions
  for (trial in 1:nTrials){
    STa = 0
    STb = 0
    # mental simulations
    for (s in 1:kapa) {
      rndNum = runif(2)
      ## Unbiased tool
      if (rndNum[1] > pBias[nfeed+1]){
        if (nfeed == 0){
          outcomeA = distSample(DistA[,1],DistA[,2],rndNum[2])
          outcomeB = distSample(DistB[,1],pEstB,rndNum[2]) }
        else {
          uniprobs = rep(1/nfeed,nfeed)
          outcomeA = distSample(ObsPay[1:nfeed,1],uniprobs,rndNum[2])
          outcomeB = distSample(ObsPay[1:nfeed,2],uniprobs,rndNum[2])
        }}
      ## Uniform tool
      else if (rndNum[1] > (2/3)*pBias[nfeed+1]){
        outcomeA = distSample(DistA[,1],rep(1/nA,nA),rndNum[2])
        outcomeB = distSample(DistB[,1],rep(1/nB,nB),rndNum[2])}
      ## Contingent Pessimism tool
      else if (rndNum[1] > (1/3)*pBias[nfeed+1]){
        if (SignMax > 0 && RatioMin < gama) {
          outcomeA = MinA
          outcomeB = MinB}
        else{
          outcomeA = distSample(DistA[,1],rep(1/nA,nA),rndNum[2])
          outcomeB = distSample(DistB[,1],rep(1/nB,nB),rndNum[2])
        }}
      ## Sign tool
      else{
        if (nfeed == 0){
          outcomeA = Range * distSample(sign(DistA[,1]),DistA[,2],rndNum[2])
          outcomeB = Range * distSample(sign(DistB[,1]),pEstB,rndNum[2])}
        else{
          uniprobs = rep(1/nfeed,nfeed);
          outcomeA = Range * distSample(sign(ObsPay[1:nfeed,1]),uniprobs,rndNum[2])
          outcomeB = Range * distSample(sign(ObsPay[1:nfeed,2]),uniprobs,rndNum[2])
        }
      }
      STa = STa + outcomeA
      STb = STb + outcomeB
    }
    STa = STa/kapa
    STb = STb/kapa

    # error term
    error = sigmat*rnorm(1); # positive values contribute to attraction to A

    # decision
    Decision[trial] = (BEVa - BEVb) + (STa - STb) + error < 0
    if ((BEVa - BEVb) + (STa - STb) + error == 0){
      Decision[trial] = sample(2,1) -1
    }

    ## Handle feedback if necessary
    if (trial >= firstFeedback){
      nfeed = nfeed +1
      pBias[nfeed+1] = beta/(beta+1+nfeed^theta)
      rndNumObs = runif(1)
      ObsPay[nfeed,1] = distSample(DistA[,1],DistA[,2],rndNumObs) # draw outcome from A
      # draw outcome from B
      if (Corr == 1){
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],rndNumObs)}
      else if (Corr == -1){
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],1-rndNumObs)}
      else{
        ObsPay[nfeed,2] = distSample(DistB[,1],DistB[,2],runif(1))
      }
      # update number of A or B "wins"
      nAwin = nAwin + (ObsPay[nfeed,1] >= ObsPay[nfeed,2])
      nBwin = nBwin + (ObsPay[nfeed,2] >= ObsPay[nfeed,1])
      sumPayB = sumPayB + ObsPay[nfeed,2]
      if (ambiguous){
        BEVb = (1-wamb)*BEVb + wamb*ObsPay[nfeed,2] # update best estimate of B's EV
        avgPayB = sumPayB/nfeed
        # update size of error in ambiguous problems
        if (subjDom != 1){
          if ((BEVa > avgPayB) & (UEVa >= UEVb)){
            sigmat = sigma * (1-nAwin/nfeed)}
          else if ((BEVa < avgPayB) & (UEVa <= UEVb)){
            sigmat = sigma * (1-nBwin/nfeed)
          }
        }
      }
    }
  }

  #compute block B-rates for this simulation
  blockSize = nTrials/nBlocks
  for (b in 1:nBlocks){
    simPred[b] = mean(Decision[((b-1)*blockSize+1):(b*blockSize)])
  }
  return(simPred)
}
