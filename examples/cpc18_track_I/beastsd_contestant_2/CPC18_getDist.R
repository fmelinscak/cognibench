CPC18_getDist = function(H,pH,L,LotShape,LotNum){
  # Extract true full distributions of an option in CPC18
  #   input is high outcome (int), its probability (double), low outcome
  #   (int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
  #   the number of outcomes in the lottery.
  #   output is a matrix with first column a list of outcomes (sorted
  #   ascending) and the second column their respective probabilities.

  if (LotShape=='-'){
    if (pH == 1){
      Dist = cbind(H, pH)}
    else{
      Dist = rbind(c(L,1-pH), c(H,pH))
    }
  }
  else { # H is multioutcome
    #compute H distribution
    if (LotShape=='Symm') {
      highDist = cbind(rep(NA,LotNum),rep(NA,LotNum))
      k = LotNum - 1
      for (i in 0:k) {
        highDist[i+1,1] = H - k/2 + i
        highDist[i+1,2] = pH*dbinom(i,k,0.5)
      }
    }
    else if ((LotShape=='R-skew') || (LotShape=='L-skew')){
      highDist = cbind(rep(NA,LotNum),rep(NA,LotNum))
      if (LotShape=='R-skew') {
        C = -1-LotNum;
        distsign = 1}
      else{
        C = 1+LotNum;
        distsign = -1
      }
      for (i in 1:LotNum){
        highDist[i,1] = H + C + distsign*2^i
        highDist[i,2] = pH/(2^i)
      }
      highDist[LotNum,2] =highDist[LotNum,2]*2
    }

    # incorporate L into the distribution
    Dist = highDist
    locb = match(L,highDist[,1])
    if (!is.na(locb)){
      Dist[locb,2] = Dist[locb,2] + (1-pH)}
    else if (pH < 1){
      Dist=rbind(Dist,c(L,1-pH))
    }
    Dist = Dist[order(Dist[,1]),]
  }
}
