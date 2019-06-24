import sciunit
from gym import spaces

### The Capabilities ###


class Interactive(sciunit.Capability):
    """
    Description:
        Requires models to respound to the environment in a interactive manner.

    For decision making cases, the model is init with the number of actions and
    number of observation.
    For associate learning models, the model is initialized with number of 
    observation.


    Observation:
        Cue/stimulus in other setting or literature. 
        Type: defined in each model.
        
    Actions:
        Type: defined in each model.
        
    Reward:
        In the decision making model setting, reward is 1 when correct action is chosen.
    """

    def predict(self):
        """Given stimulus, model should return the function for the calculation
        of log probability density function or log probability mass function.
        """
        raise NotImplementedError("Must implement predict.")
    
    def reset(self):
        """Reset the hidden state of the model to initial state.
        """
        raise NotImplementedError("Must implement reset.")

    def update(self):
        """Given stimulus, rewards and action, the model should updates its 
        hidden state. Also named evolution function in some packages.
        """
        raise NotImplementedError("Must implement update.")

    def act(self):
        """For decision making, return the action taken by the model,
        Associative learning models returns the predicted value.
        Also named observation function in some packages.
        """
        raise NotImplementedError("Must implement act")

class DiscreteAction(sciunit.Capability):
    def _set_action_space(self, *args, **kwargs):
        return spaces.Discrete(*args, **kwargs)

class ContinuousAction(sciunit.Capability):
    def _set_action_space(self, *args, **kwargs):
        return spaces.Box(*args, **kwargs)

class MultibinObsevation(sciunit.Capability):
    def _set_observation_space(self, *args, **kwargs):
        return spaces.MultiBinary(*args, **kwargs)

class DiscreteObservation(sciunit.Capability):
    def _set_observation_space(self, *args, **kwargs):
        return spaces.Discrete(*args, **kwargs)
