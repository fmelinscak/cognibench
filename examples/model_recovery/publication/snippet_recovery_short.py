import cognibench as cnb

# Define models
n_action, n_obs = 2, 2
model_list = [
    cnb.RWModel(n_action=n_action, n_obs=n_obs),
    cnb.CKModel(n_action=n_action, n_obs=n_obs),
    cnb.RWCKModel(n_action=n_action, n_obs=n_obs),
    cnb.NWSLSModel(n_action=n_action, n_obs=n_obs),
]
# Define environment and test
env = cnb.BanditEnv(p_dist=[0.2, 0.8])
score_cls = cnb.partialclass(cnb.AICScore, min_score=0, max_score=1000)
test_cls = cnb.partialclass(cnb.InteractiveTest, score_type=score_cls)
# Run many times and pick best models
for _ in range(N_SIMULATIONS):
    _, score_matrix = cnb.model_recovery(model_list, env, test_cls)
    sm_arr = score_matrix_to_numpy(score_matrix)
    min_indices = numpy.argmin(sm_arr, axis=0)
    confusion_matrix[min_indices, numpy.arange(N_MODELS)] += 1
confusion_matrix /= numpy.sum(confusion_matrix, axis=0)
