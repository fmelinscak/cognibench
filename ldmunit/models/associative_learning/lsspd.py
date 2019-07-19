import numpy as np
import gym
from gym import spaces
from scipy import stats
from ldmunit.models import CAMO
from ldmunit.capabilities import Interactive, LogProbModel
from ldmunit.utils import is_arraylike


class LSSPDModel(CAMO, Interactive, LogProbModel):
    """
    LSSPD model implementation.
    """
    # TODO: what is the name of this model?
    name = 'LSSPD'

    def __init__(self, *args, w, alpha, b0, b1, sigma, mix_coef, eta, kappa, **kwargs):
        """
        Parameters
        ----------
        w : float or array-like
            Initial value of weight vector w. If float, then all elements of the
            weight vector is set to this value. If array-like, must have the same
            length as the dimension of the observation space.

        alpha : float or array-like
            Initial value of associability vector alpha. If float, then all elements of the
            weight vector is set to this value. If array-like, must have the same
            length as the dimension of the observation space.

        b0 : float
            Intercept used when computing the mean of normal distribution from reward.

        b1 : float
            Slope used when computing the mean of the normal distribution from reward.

        sigma : float
            Standard deviation of the normal distribution used to generate observations.
            Must be nonnegative.

        mix_coef : float
            Mixing coefficient used in the convex combination of weight and associability vectors.
            Must be in [0, 1] range.

        eta : float
            Learning rate for alpha updates. Must be nonnegative.

        kappa : float
            Learning rate for w updates. Must be nonnegative.

        Other Parameters
        ----------------
        **kwargs : any type
            All the mandatory keyword-only arguments required by :class:`ldmunit.models.associative_learning.base.CAMO` must also be
            provided during initialization.
        """
        assert sigma >= 0, 'sigma must be nonnegative'
        assert mix_coef >= 0 and mix_coef <= 1, 'mix_coef must be in range [0, 1]'
        assert eta >= 0, 'eta must be nonnegative'
        assert kappa >= 0, 'kappa must be nonnegative'
        paras = {
            'w': w,
            'alpha': alpha,
            'b0': b0,
            'b1': b1,
            'sigma': sigma,
            'mix_coef': mix_coef,
            'eta': eta,
            'kappa': kappa
        }
        super().__init__(paras=paras, **kwargs)
        if is_arraylike(w):
            assert len(w) == self.n_obs, 'w must have the same length as the dimension of the observation space'
        if is_arraylike(alpha):
            assert len(alpha) == self.n_obs, 'alpha must have the same length as the dimension of the observation space'

    def reset(self):
        """
        Reset the hidden state to its default value.
        """
        w = self.paras['w'] if 'w' in self.paras else 0
        alpha = self.paras['alpha'] if 'alpha' in self.paras else 0

        if is_arraylike(w):
            w = np.array(w)
        else:
            w = np.full(self.n_obs, w)

        if is_arraylike(alpha):
            alpha = np.array(alpha)
        else:
            alpha = np.full(self.n_obs, alpha)

        self.hidden_state = {'w': w, 'alpha': alpha}

    def observation(self, stimulus):
        """
        Get the reward random variable for the given stimulus.

        Parameters
        ----------
        stimulus : :class:`np.ndarray`
            Single stimulus from the observation space.

        Returns
        -------
        :class:`scipy.stats.rv_continuous`
            Normal random variable with mean equal to reward and
            standard deviation equal to sigma model parameter.
        """
        assert self.observation_space.contains(stimulus)

        b0 = self.paras['b0']
        b1 = self.paras['b1']
        sd_pred = self.paras['sigma']
        mix_coef = self.paras['mix_coef']

        w_curr = self.hidden_state['w']
        alpha = self.hidden_state['alpha']

        # Predict response
        mu_pred = b0 + b1 * np.dot(stimulus, (mix_coef * w_curr + (1 - mix_coef) * alpha))

        rv = stats.norm(loc=mu_pred, scale=sd_pred)
        rv.random_state = self.seed

        return rv

    def predict(self, stimulus):
        """
        Predict the log-pdf over the continuous action space by using the
        given stimulus as input.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        Returns
        -------
        method
            :py:meth:`scipy.stats.rv_continuous.logpdf` method over the continuous action space.
        """
        return self.observation(stimulus).logpdf

    def act(self, stimulus):
        """
        Return an action for the given stimulus.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        Returns
        -------
        float
            An action from the continuous action space.
        """
        return self.observation(stimulus).rvs()

    def _predict_reward(self, stimulus):
        assert self.observation_space.contains(stimulus)
        w_curr = self.hidden_state['w']
        rhat = np.dot(stimulus, w_curr.T)
        return rhat

    def update(self, stimulus, reward, action, done):
        """
        Update the hidden state of the model based on input stimulus, action performed
        by the model and reward.

        Parameters
        ----------
        stimulus : array-like
            A stimulus from the multi-binary observation space for this model. For
            example, `[0, 1, 1]`.

        reward : float
            The reward for the action.

        done : bool
            If True, do not update the hidden state.
        """
        assert self.action_space.contains(action)
        assert self.observation_space.contains(stimulus)

        eta = self.paras['eta']  # Proportion of pred. error. in the updated associability value
        kappa = self.paras['kappa']  # Fixed learning rate for the cue weight update

        w_curr = self.hidden_state['w']
        alpha = self.hidden_state['alpha']

        rhat = self._predict_reward(stimulus)

        if not done:
            delta = reward - rhat

            w_curr += kappa * delta * alpha * stimulus  # alpha, stimulus size: (n_obs,)

            # if stimulus[i] = 1
            # alpha[i] = eta * abs(pred_err) + (1 - eta) * alpha[i]
            # or
            # alpha[i] -= eta * alpha[i]
            # alpha[i] += eta * abs(pred_err)
            alpha -= eta * np.multiply(alpha, stimulus)
            alpha += eta * abs(delta) * stimulus
            np.clip(alpha, a_min=0, a_max=1, out=alpha)  # Enforce upper bound on alpha

            self.hidden_state['w'] = w_curr
            self.hidden_state['alpha'] = alpha

        return w_curr, alpha
