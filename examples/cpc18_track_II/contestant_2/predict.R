predict_R <- function(stimuli) {
    df <- data.frame("SubjID" = stimuli[,1], "GameID" = stimuli[,6], "block" = stimuli[,19], "B" = stimuli[,19]*0)
    df$new.GameID = paste(df$GameID,df$block, sep = ".")
    df$new.GameID = factor(df$new.GameID, levels=train_levels)
    df$GameID = df$new.GameID
    df = df[ , !(names(df) %in% drops)]

    x = lapply(df[,sapply(df, is.factor),drop=FALSE], contrasts, contrasts=FALSE)
    features = model.matrix(B~., data = df, contrasts.arg = x)[,-1]
    pred = predict(clf, features)

    return(pred)
}
