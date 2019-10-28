distSample = function( Numbers, Probabilities, rndNum ){
  # Sampling a single number from a discrete distribution
  #   The possible Numbers in the distribution with their resective
  #   Probabilities. rndNum is a randomly drawn probability
  #
  #   Conditions on Input (not checked):
  #   1. Numbers and Probabilites correspond one to one (i.e. first number is
  #   drawn w.p. first probability etc)
  #   2. rndNum is a number between zero and one
  #   3. Probabilites is a probability vector

  cumProb = 0
  sampledInt = 0
  while (rndNum > cumProb){
    sampledInt = sampledInt +1
    cumProb = cumProb + Probabilities[sampledInt]
  }
  return(Numbers[sampledInt])
}
