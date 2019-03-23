import sciunit


### The Capabilities ###

# Each capability is implemented separately, although many rely on 40Hz drive
# This increases the computating time, but makes the implementation simpler and more clear

class ProducesLoglikelihood(sciunit.Capability):

    def produce_loglikelihood(self):
        """Produce a number."""
        raise NotImplementedError("Must implement produce_loglikelihood.")
