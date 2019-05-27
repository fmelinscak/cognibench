from sciunit import Test
from .scores import NLLScore, BICScore, AICScore
from .capabilities import ProducesLoglikelihood

class NLLTest(Test):

    required_capabilities = (ProducesLoglikelihood, )
    score_type = NLLScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        return model.produce_loglikelihood(stimuli, rewards)

    def compute_score(self, observation, prediction):
        score = NLLScore.compute(observation, prediction)
        return score

class BICTest(Test):

    required_capabilities = (ProducesLoglikelihood, )
    score_type = BICScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        return model.produce_loglikelihood(stimuli, rewards), model.get_n_obs(rewards)

    def compute_score(self, observation, prediction):
        score = BICScore.compute(observation, prediction)
        return score

class AICTest(Test):

    required_capabilities = (ProducesLoglikelihood, )
    score_type = AICScore

    def generate_prediction(self, model):
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        return model.produce_loglikelihood(stimuli, rewards)

    def compute_score(self, observation, prediction):
        score = AICScore.compute(observation, prediction)
        return score