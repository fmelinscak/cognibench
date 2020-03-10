import cognibench as cnb

# Define models
model_list = [
    cnb.MultiRwNormModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiKrwNormModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiBetaBinomialModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiLSSPDModel(n_subj=N_SUBJECTS, n_obs=4),
]
# Define tests
score_cls = cnb.partialclass(cnb.AICScore, min_score=0, max_score=1000)
test_list = []
for test_name, path in zip(test_names, model_list):
    # simulate data with two stages and two cues (not shown here)
    obs = get_simulation_data(model)
    test_list.append(
        cnb.InteractiveTest(
            name=test_name, multi_subject=True, observation=obs, score_type=score_cls
        )
    )
# Define suite and judge
suite = sciunit.TestSuite(test_list, name="Associative learning suite")
score_matrix = suite.judge(model_list)
