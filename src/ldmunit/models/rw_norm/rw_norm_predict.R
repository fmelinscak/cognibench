rw_norm_predict <- function(X, r, alpha, sigma, b0, b1, w0) {
    
    n_trials = nrow(X)
    n_features = ncol(X)
    
    w <- matrix(data=NA,nrow=n_trials + 1,ncol=n_features)
    w[1,] <- w0
    
    mu_pred <- matrix(data=NA, nrow=n_trials)
    sd_pred <- matrix(data=sigma, nrow=n_trials)
    
    for (i in 1:n_trials) {
        w_curr = w[i,]
        x_curr = X[i,]
        
        # Generate outcome prediction
        rhat = x_curr %*% w_curr
    
        # Predict response
        mu_pred[i] = b0 + b1*rhat
        
        #Calculate prediction error based on observed outcome
        pred_err = r[i] - rhat
        
        # Update weights of active cues
        w[i+1,] = w_curr + (alpha*pred_err) %*% x_curr
    }
    res <- list("mu_pred" = mu_pred, "sd_pred" = sd_pred)
    # print(w[0:5,])
    return(res)
}