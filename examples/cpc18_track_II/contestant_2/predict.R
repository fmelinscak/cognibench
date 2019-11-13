predict <- function(stimuli) {
    df <- data.frame("SubjID" = stimuli[,1], "GameID" = stimuli[,6], "block" = stimuli[,19], "B" = actions)
    df$new.GameID = paste(df$GameID,df$block, sep = ".")
    df$new.GameID = factor(df$new.GameID)
    df$GameID = df$new.GameID
    df = df[ , !(names(df) %in% drops)]

    features = model.matrix(B~., data = df, contrasts.arg = lapply(df[,sapply(df, is.factor) ], contrasts, contrasts=FALSE))[,-1]
    pred = predict(model, features)

    return(pred)
}
