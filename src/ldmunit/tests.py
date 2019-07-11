from sciunit import Test
from sciunit import scores
from .capabilities import Interactive
from .models.utils import loglike

class NLLTest(Test):
    """Calculate negative loglikelihood of a model."""
    required_capabilities = (Interactive, )
    score_type = scores.FloatScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        res = 0
        if hasattr(model, 'models'):
            for m, s, r, a in zip(model.models, stimuli, rewards, actions):
                res += loglike(m, s, r, a)
        else:
            res += loglike(model, stimuli, rewards, actions)

        return - res

    def compute_score(self, observation, prediction):
        return self.score_type(prediction)


class AICTest(Test):
    """Calculate AIC score of a model on given data."""
    required_capabilities = (Interactive, )
    score_type = scores.FloatScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        res = 0
        if hasattr(model, 'models'):
            for m, s, r, a in zip(model.models, stimuli, rewards, actions):
                res -= 2 * loglike(m, s, r, a)
                res += 2 * len(m.paras)
        else:
            res -= loglike(model, stimuli, rewards, actions)
            res += 2 * len(model.paras)

        return res

    def compute_score(self, observation, prediction):
        return self.score_type(prediction)

class BICTest(Test):
    """Calculate BIC score of a model on given data."""
    required_capabilities = (Interactive, )
    score_type = scores.FloatScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        res = 0
        if hasattr(model, 'models'):
            for m, s, r, a in zip(model.models, stimuli, rewards, actions):
                res -= 2 * loglike(m, s, r, a)
                res += len(s) * len(m.paras)
        else:
            res -= loglike(model, stimuli, rewards, actions)
            res += len(stimuli) * len(model.paras)

        return res

    def compute_score(self, observation, prediction):
        return self.score_type(prediction)