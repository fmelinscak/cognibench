import sciunit


### The Capabilities ###


class Interactive(sciunit.Capability):
    """
    Description:
        Requires models to respound to the environment in a interactive manner.

    Observation: 
        Type: defined in each model.
        
    Actions:
        Type: defined in each model.
        
    Reward:
        In the decision making model setting, reward is 1 when correct action is chosen.
    """

    def predict(self):
        """Return pmf/pdf based on stimulus (observation in AI Gym)."""
        raise NotImplementedError("Must implement predict.")
    
    def reset(self):
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
        raise NotImplementedError("Must implement reset.")

    def update(self):
        """Update model's state given stimulus (observation in AI Gym), reward, action in the environment."""
        raise NotImplementedError("Must implement update.")

    def act(self):
        raise NotImplementedError("Must implement act")