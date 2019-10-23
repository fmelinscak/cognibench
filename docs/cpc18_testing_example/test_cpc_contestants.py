from ldmunit.models.utils import multi_from_single_interactive
from testing_definitions import MSETest, BEASTsdModel
import numpy as np
import pandas as pd
import sciunit


if __name__ == '__main__':
    # Read the data and construct feature vectors
    data = pd.read_csv('CPC18_EstSet.csv')
    n_probs = data.shape[0]
    feature_names = ['Ha', 'pHa', 'La', 'LotShapeA', 'LotNumA', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB', 'Amb', 'Corr']
    features = []
    for prob_idx in range(n_probs):
        row = []
        for name in feature_names:
            val = data[name][prob_idx]
            row.append(val)
        features.append(row)

    # Put the data into the format expected by interactive tests
    actions = data[['B.1', 'B.2', 'B.3', 'B.4', 'B.5']].values
    rewards = [-1] * len(features)
    obs = {'rewards': [rewards], 'actions': [actions], 'stimuli': [features]}

    # Create model objects from models stored in folders submitted by
    # each contestant.
    contestant_ids = [0, 1, 2]
    mse_test = MSETest(name='Aggregate MSE Test', observation=obs)
    MultiBEAST = multi_from_single_interactive(BEASTsdModel)
    model_list = []
    for cid in contestant_ids:
        import_base_path = 'beastsd_contestant_{}'.format(cid)
        multi_beast = MultiBEAST([{'import_base_path': import_base_path}])
        multi_beast.name = 'Contestant_{}'.format(cid)
        model_list.append(multi_beast)
    
    # Run the tests and score every contestant
    mse_suite = sciunit.TestSuite([mse_test], name="MSE suite")
    mse_suite.judge(model_list)
