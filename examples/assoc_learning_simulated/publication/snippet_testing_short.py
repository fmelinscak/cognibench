import cognibench as cnb

# Define tests
score_cls = cnb.partialclass(cnb.AICScore, min_score=0, max_score=1000)
test_list = []
for test_name, path in test_names_and_paths:
    obs = get_simulation_data(path, N_SUBJECTS, True)
    test_list.append(
        cnb.InteractiveTest(
            name=f"{test_name}",
            multi_subject=True,
            observation=obs,
            score_type=score_cls,
        )
    )
# Define models
model_list = [
    cnb.MultiRwNormModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiKrwNormModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiBetaBinomialModel(n_subj=N_SUBJECTS, n_obs=4),
    cnb.MultiLSSPDModel(n_subj=N_SUBJECTS, n_obs=4),
]
# Define suite and judge
suite = sciunit.TestSuite(test_list, name="Associative learning suite")
score_matrix = suite.judge(model_list)
