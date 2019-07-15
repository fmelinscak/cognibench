from .utils import partialclass
from sciunit import Test
from .scores import SmallerBetterScore
from .capabilities import Interactive, LogProbModel

def _test_single_model(model, stimuli, rewards, actions):
    """
    _test_single_model is a private utility function which that trains
    a model on a list of stimulus, reward, action triples. The list of
    predictions after each predict step is returned.
    
    Parameters
    ----------
    model:    Model object used to iteratively predict the stimuli and
              update using actions and rewards.

    stimuli:  List of stimuli.
    rewards:  List of rewards.
    actions:  List of actions.

    Returns
    -------
    res:      List of predictions in the same order as the given triples.
    """
    predictions = []
    model.reset()

    for s, r, a in zip(stimuli, rewards, actions):
        predictions.append(model.predict(s))
        model.update(s, r, a, False)

    return predictions


class InteractiveTest(Test):
    """
    Perform interactive tests by feeding the input samples (stimuli) one at a
    time.
    """
    required_capabilities = (Interactive, )

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        if hasattr(model, 'models'):
            models = model.models
        else:
            models = [model]

        predictions = []
        for m, s, r, a in zip(models, stimuli, rewards, actions):
            predictions.append(_test_single_model(m, s, r, a))

        return predictions


class NLLTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Negative log-likelihood (NLL) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def compute_score(self, observation, prediction):
        actions = observation['actions']
        score = 0
        n_models = len(actions)
        for i in range(n_models):
            for act, logprob in zip(actions[i], prediction[i]):
                score -= logprob(act)
        return self.score_type(score)


class AICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Akaike Information Criterion (AIC) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def generate_prediction(self, model):
        # save variables necessary to compute score
        if hasattr(model, 'models'):
            models = model.models
        else:
            models = [model]
        self.n_model_params = [len(m.paras) for m in models]

        return super().generate_prediction(model)

    def compute_score(self, observation, prediction):
        actions = observation['actions']
        score = 0
        n_models = len(actions)
        for i in range(n_models):
            for act, logprob in zip(actions[i], prediction[i]):
                score -= logprob(act)

        score += 2 * sum(self.n_model_params)
        return self.score_type(score)


class BICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Bayesian Information Criterion (BIC) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def generate_prediction(self, model):
        # save variables necessary to compute score
        stimuli = self.observation['stimuli']
        if hasattr(model, 'models'):
            models = model.models
        else:
            models = [model]
        self.n_model_params = [len(m.paras) for m in models]
        self.n_samples = [len(s) for s in stimuli]

        return super().generate_prediction(model)

    def compute_score(self, observation, prediction):
        actions = observation['actions']
        score = 0
        n_models = len(actions)
        for i in range(n_models):
            for act, logprob in zip(actions[i], prediction[i]):
                score -= logprob(act)

        score += sum(p * q for p, q in zip(self.n_model_params, self.n_samples))
        return self.score_type(score)
