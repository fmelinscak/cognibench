import sciunit


### The Capabilities ###


class ProducesLoglikelihood(sciunit.Capability):

    def produce_loglikelihood(self):
        """Produces the loglikelihood function for agent's actions."""
        raise NotImplementedError("Must implement produce_loglikelihood.")
