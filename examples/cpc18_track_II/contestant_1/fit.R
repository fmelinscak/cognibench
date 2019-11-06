fit <- function(stimuli, actions) {
    df <- data.frame("SubjID" = stimuli[,1], "GameID" = stimuli[,6], "block" = stimuli[,19], "B" = actions)
    aggreg <- aggregate(B~GameID+block, data=df, FUN=mean)
    avgBinTrain <<- data.matrix(aggreg)
    overallMean <<- mean(avgBinTrain[,3])
}
