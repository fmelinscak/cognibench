# CogniBench
CogniBench is a software framework for benchmarking cognitive models.
It is implemented as a free, open source package in Python, but it can readily be used for validating models implemented in any language for which there is a Python interface (such as R or Matlab/Octave).
CogniBench builds upon [SciUnit](https://github.com/scidash/sciunit) - a domain-agnostic framework for validating scientific models,  and [OpenAI Gym](https://github.com/openai/gym) - a library for developing and testing artificial agents.
For a short introduction to CogniBench structure, please refer to documentation notebooks under the `docs` folder. For detailed examples of how one can use CogniBench you can refer to example testing and simulation
scripts under the `examples` folder.

## Installation
<!-- TODO: Update pip installation instructions when the package is to be published on PyPI. -->
You can install CogniBench by downloading or cloning the repository, and running the following command:

```bash
pip install cognibench-path
```
where `cognibench-path` is the path to the top-level directory of the unpacked/cloned CogniBench package (i.e., the directory that contains the `setup.py` file).

If you wish to contribute to the development, you can clone this repository, and create the [conda](https://docs.conda.io/en/latest/) development environment with CogniBench using the following commands executed at the package top-level directory:
```bash
conda env create -f environment.yml
conda activate cognibench
python setup.py install
```

## Short usage example
Here is a short snippet describing how you can test several models against multiple sets of experimental observations using CogniBench.

```python
import cognibench.models.decision_making as decision_models
from cognibench.testing import InteractiveTest
from cognibench.scores import AccuracyScore, PearsonCorrelationScore
from sciunit import TestSuite

# observations is a dictionary with keys such as 'stimuli', 'rewards', etc.
observations, obs_dim, action_dim = read_data(observation_path)
# define the list of models to test
model_list = [
    decision_models.RWCKModel(n_action=action_dim, n_obs=obs_dim, seed=42),
    decision_models.NWSLSModel(n_action=action_dim, n_obs=obs_dim, seed=42),
]
# define the list of test cases
test_list = [
    InteractiveTest(observation=observations, score_type=AccuracyScore, name='Accuracy Test'),
    InteractiveTest(observation=observations, score_type=PearsonCorrelationScore, name='Correlation Test'),
]
# combine in a suite and run
test_suite = TestSuite(test_list, name='Test suite')
test_suite.judge(model_list)
```

## Main features of CogniBench

### Interactive tests
Testing certain models should be performed in an interactive manner. Instead of presenting all the stimuli at once, test
samples are inputted one at a time, while observing the actions of the model being tested. CogniBench formalizes this
notion in `InteractiveTest` test class and `Interactive` model capability.

In addition to interactive tests, CogniBench also implements the common way of testing models against a batch of samples
(`BatchTest` and `BatchTestWithSplit`) in case you don't need the interactive testing logic.

### SciUnit and OpenAI Gym interaction
In the SciUnit framework, models are tagged with capabilities which define the tests a model can possibly take.
CogniBench combines this idea with action and observation spaces from OpenAI Gym library. Therefore, a model also specifies
against which environments it can be simulated against in addition to the tests it can take.

### Support for both single- and multi-subject models
Some models operate on a single subject at a time (single-subject models) whereas others can operate on multiple subjects
at the same time (multi-subject models). CogniBench supports multi-subject models by assuming the model implementation of
required interface functions take the subject index as the first argument. The testing interface defined by `CNBTest`
class can seamlessly work on both single- and multi-subject models. In addition, we provide a simple utility function
to convert single-subject model classes deriving from `CNBModel` to multi-subject classes.

### Data simulation
CogniBench provides utility functions to simulate agents and/or models against matching environments to generate
stimuli, action and reward triplets. These functions support both single-subject and multi-subject models.

### Implementation of common experimental tasks
CogniBench offers `model_recovery` and `param_recovery` functions that you can use to perform these common auxiliary modeling tasks.

### Agent and model Separation
CogniBench distinguishes between agents (`CNBAgent` base class) and models (`CNBModel` base class). An agent can
interact with an environment through `act` and `update` methods, and can only function when its parameters are set to
given values. In contrast, a model represents a specific way of fitting parameters for an agent (`fit`) and predicting
the probability distribution over the action space (`predict`). The models we provide in CogniBench are implemented by
taking this distinction into consideration; however, CogniBench has the flexibility to support models that don't care
about this distinction.

### Associative learning agent and model implementations
We provide example implementations for several simple associative learning agents and models. These models also demonstrate
how to satisfy the interfaces required by interactive tests that require log probability distributions as predictions.
Currently implemented associative learning models are
* Random responding
* Beta-binomial
* Rescorla-Wagner
* Kalman Rescorla-Wagner
* LSSPD (Rescorla-Wagner-Pearce-Hall)

### Decision making agent and model implementations
Similarly, we also provide example implementations for several simple decision making agents and models. Currently
implemented decision making models are
* Random responding
* Rescorla-Wagner Choice Kernel
* Rescorla-Wagner
* Choice Kernel
* Noisy-win-stay-lose-shift

## Documentation
We provide a series of Jupyter notebooks that you can use as an introduction to CogniBench:
<!-- TODO: Below links should point to github because then they will work from readthedocs and so on. -->
<!-- TODO: change the name of the repository to cognibench -->
* [Chapter 1: Introduction](docs/ch01_introduction.ipynb)
* [Chapter 2: Tests](docs/ch02_tests.ipynb)
* [Chapter 3: Models](docs/ch03_models.ipynb)
* [Chapter 4: Complete Example](docs/ch04_complete_example.ipynb)
* [Chapter 5: Simulation](docs/ch05_simulation.ipynb)
* [Chapter 6: Tasks](docs/ch06_tasks.ipynb)

<!-- TODO: API reference link must be https (readthedocs) -->
<!-- Additionally, you can browse our [API reference](TODO) to get more information about certain functions,
classes, etc. you want to use. -->

##### Small note to developers
If you are going to use the development version and want to run notebooks, you should install CogniBench inside the conda
environment. See the section on installing CogniBench.

## Examples
We provide multiple examples of using CogniBench as a tool to test models, simulate data and perform experimental tasks.
These are very useful to get acquainted with how to use CogniBench.  Please refer to readme file under `examples/` folder
for further information.

## Tests
We use built-in `unittest` module for testing CogniBench. To perform checks, clone this repository and type

```bash
./test.sh
```

## Information for developers
If you want to extend CogniBench, you need to use the development environment and `conda`. Please follow the conda
installation instructions in how to install section and then continue here.

We use [`black`](https://github.com/psf/black) for code-formatting and [`pre-commit`](https://pre-commit.com/) for ensuring high quality code.  To enable these tools simply run

```bash
pre-commit install
```

The next time you try to commit, all the required tools and hooks will be downloaded (and cached) and checks will be
performed on your code.

### Generating local documentation
After enabling the development environment, you can generate a local version of the documentation by running

```bash
cd docs/sphinx
make html
```

Afterwards, you can browse the local documentation by opening `docs/sphinx/_build/index.html`.

### Installing a local development version
After implementing some changes, you can install the modified version of CogniBench to your local system by running

```bash
python setup.py install
```

Then, every time you import CogniBench, this modified version will be imported.

## License
<!-- TODO: LICENSE link must be https (github) -->
CogniBench is distributed under [MIT license](https://opensource.org/licenses/MIT).
See the [LICENSE](LICENSE) file for the exact terms and conditions.
