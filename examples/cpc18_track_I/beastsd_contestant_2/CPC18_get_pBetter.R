get_pBetter  = function( DistX, DistY, corr, accuracy = 10000 ) {
  #Return probability that a value drawn from DistX is strictly larger than one drawn from DistY
  # Input: 2 discrete distributions which are set as matrices of 1st column
  # as outcome and 2nd its probability; correlation between the distributions;
  # level of accuracy in terms of number of samples to take from distributions
  # Output: the estimated probability that X generates value strictly larger than Y, and
  # the probability that Y generates value strictly larger than X

  nXbetter = 0
  nYbetter = 0
  for (j in 1:accuracy){
    rndNum = runif(2)
    sampleX = distSample(DistX[,1],DistX[,2],rndNum[1])
    if (corr == 1){
      sampleY = distSample(DistY[,1],DistY[,2],rndNum[1])}
    else if (corr == -1) {
      sampleY = distSample(DistY[,1],DistY[,2],1-rndNum[1])}
    else {
      sampleY = distSample(DistY[,1],DistY[,2],rndNum[2])
    }
    nXbetter = nXbetter + as.numeric(sampleX > sampleY)
    nYbetter = nYbetter + as.numeric(sampleY > sampleX)
  }
  pXbetter = nXbetter/accuracy
  pYbetter = nYbetter/accuracy
  return(list(pXbetter,pYbetter))
}
