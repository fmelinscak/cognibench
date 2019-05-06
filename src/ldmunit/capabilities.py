import sciunit


### The Capabilities ###


class ProducesLoglikelihood(sciunit.Capability):

    def produce_loglikelihood(self):
        """Produces the loglikelihood function for agent's actions."""
        raise NotImplementedError("Must implement produce_loglikelihood.")

class Trainable(sciunit.Capability):

    def train_with_observations(self):
        """Fit the parameters of the models."""
        raise NotImplementedError("Must implement train_with_observations.")