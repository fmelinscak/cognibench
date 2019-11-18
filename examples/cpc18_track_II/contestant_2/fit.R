fit <- function(stimuli, actions) {
    df <- data.frame("SubjID" = stimuli[,1], "GameID" = stimuli[,6], "block" = stimuli[,19], "B" = actions)
    df$new.GameID = paste(df$GameID,df$block, sep = ".")
    df$new.GameID = factor(df$new.GameID)
    train_levels <<- levels(df$new.GameID);
    df$GameID = df$new.GameID
    df = df[ , !(names(df) %in% drops)]

    x = lapply(df[,sapply(df, is.factor),drop=FALSE], contrasts, contrasts=FALSE)
    features = model.matrix(B~., data = df, contrasts.arg = x)[,-1]
    target = df$B
    clf <<- FM.train(features, target, c(1,5), iter=10, regular=c(0.001, 0.05), intercept=TRUE)
}
