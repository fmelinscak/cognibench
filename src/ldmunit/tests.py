from .utils import partialclass
from sciunit import Test
from .scores import SmallerBetterScore
from .capabilities import Interactive, LogProbModel

def _test_multimodel(multimodel, stimuli, rewards, actions):
    """
    _test_multimodel is a private utility function which trains
    a multi-subject model on a list of stimulus, reward, action
    triples for each subject. The list of predictions after each
    predict step is returned.
    
    Parameters
    ----------
    multimodel : sciunit.Model and capabilities.Interactive
        Multi-subject model. If you have a single-subject model
        that works on data for one subject at a time, use
        models.utils.multi_from_single to get a multi-subject
        model that works on a single-subject at a time.

    stimuli : list of list
        List of subject-specific stimuli. Each element of this list
        must contain all the stimuli for the corresponding subject as a list.

    rewards : list of list
        List of subject-specific rewards. Each element of this list
        must contain all the rewards for the corresponding subject as a list.

    actions : list of list
        List of subject-specific rewards. Each element of this list
        must contain all the rewards for the corresponding subject as a list.

    Returns
    -------
    list of list
        List of predictions. Each element of this list contains
        all the predictions for the corresponding subject as a list.
    """
    predictions = []

    for subject_idx, (subject_stimuli, subject_rewards, subject_actions) in enumerate(zip(stimuli, rewards, actions)):
        multimodel.reset(subject_idx)
        subject_predictions = []
        for s, r, a in zip(subject_stimuli, subject_rewards, subject_actions):
            subject_predictions.append(multimodel.predict(subject_idx, s))
            multimodel.update(subject_idx, s, r, a, False)
        predictions.append(subject_predictions)

    return predictions


def _neg_loglikelihood(actions, predictions):
    """
    Compute negative log-likelihood of a multimodel using a collection of
    subject-specific true action and model prediction lists. Each prediction
    list must contain a series of logpdf or logpmf functions.

    Parameters
    ----------
    actions : list of list
        List of subject-specific actions. Each element must be a list
        containing a series of actions.
    predictions : list of list
        List of subject-specific predictions. Each element must be a list
        containing a series of predictions as logpdf or logpmf.

    Returns
    -------
    float
        Negative log-likelihood of the whole multi-subject model on the
        given action and prediction data. It is calculated as the sum of
        individual log probabilities for every action-prediction pairs.
    """
    neg_loglike = 0
    n_subjects = len(actions)
    for subject_idx in range(n_subjects):
        for act, logprob in zip(actions[subject_idx], predictions[subject_idx]):
            neg_loglike -= logprob(act)
    return neg_loglike


class InteractiveTest(Test):
    """
    Perform interactive tests by feeding the input samples (stimuli) one at a
    time. This class is not intended to be used directly since it does not
    specify how the score should be computed. In order to create concrete
    interactive tests, create a subclass and specify how the score should be
    computed.

    See Also
    --------
    NLLTest, AICTest, BICTest for examples of concrete interactive test classes
    """
    required_capabilities = (Interactive, )

    def generate_prediction(self, multimodel):
        """
        Generate predictions from a multi-subject model one at a time.

        Parameters
        ----------
        multimodel : sciunit.Model and capabilities.Interactive
            Multi-subject model

        Returns
        -------
        list of list
            Predictions

        See Also
        --------
        _test_multimodel
        """
        stimuli = self.observation['stimuli']
        rewards = self.observation['rewards']
        actions = self.observation['actions']

        predictions = _test_multimodel(multimodel, stimuli, rewards, actions)
        return predictions


class NLLTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Negative log-likelihood (NLL) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def compute_score(self, observation, prediction):
        """
        Compute the negative log-likelihood score from observations and predictions

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        SmallerBetterScore
            Negative log-likelihood.
        """
        nll = _neg_loglikelihood(observation['actions'], prediction)
        return self.score_type(nll)


class AICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Akaike Information Criterion (AIC) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def generate_prediction(self, multimodel):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        InteractiveTest.generate_prediction
        """
        # save variables necessary to compute score
        self.n_model_params = [len(m.paras) for m in multimodel.subject_models]

        return super().generate_prediction(multimodel)

    def compute_score(self, observation, prediction):
        """
        Compute the Akaike Information Criterion score from observations, predictions
        and, model and input specific parameters stored during generate_prediction.

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        SmallerBetterScore
            AIC
        """
        nll = _neg_loglikelihood(observation['actions'], prediction)
        regularizer = 2 * sum(self.n_model_params)
        return self.score_type(nll + regularizer)


class BICTest(InteractiveTest):
    """
    Perform interactive test on models that produce a log pdf/pmf as their
    predictions. Bayesian Information Criterion (BIC) function is used as the score.
    """
    score_type = partialclass(SmallerBetterScore, min_score=0, max_score=1000)
    required_capabilities = InteractiveTest.required_capabilities + (LogProbModel,)

    def generate_prediction(self, multimodel):
        """
        This method simply calls the parent method for the actual functionality
        after storing some necessary variables to compute the score later.

        See Also
        --------
        InteractiveTest.generate_prediction
        """
        # save variables necessary to compute score
        stimuli = self.observation['stimuli']
        self.n_model_params = [len(m.paras) for m in multimodel.subject_models]
        self.n_samples = [len(s) for s in stimuli]

        return super().generate_prediction(multimodel)

    def compute_score(self, observation, prediction):
        """
        Compute the Bayesian Information Criterion score from observations, predictions
        and, model and input specific parameters stored during generate_prediction.

        Parameters
        ----------
        observation : dict
            Dictionary storing the actions in 'actions' key.

        prediction : list of list
            List of subject-specific predictions. Each element is a list and
            stores the predictions for the corresponding subject.

        Returns
        -------
        SmallerBetterScore
            BIC
        """
        nll = _neg_loglikelihood(observation['actions'], prediction)
        regularizer = sum(p * q for p, q in zip(self.n_model_params, self.n_samples))
        return self.score_type(nll + regularizer)
