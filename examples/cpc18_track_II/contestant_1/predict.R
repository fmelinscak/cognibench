predict <- function(stimulus) {
    print(stimulus)
    game_id <- stimulus[6]
    block_id <- stimulus[19]
    mask <- (avgBinTrain[,1] == game_id) & (avgBinTrain[,2] == block_id)
    ans <- overallMean
    n_valid_rows = sum(mask)
    if (n_valid_rows > 0) {
        stopifnot(n_valid_rows == 1)
        ans <- avgBinTrain[mask, 3]
    }
    print(ans)
    return(ans)
}
