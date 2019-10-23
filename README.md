# LDMUnit
LDMUnit is a framework for test and validation
of learning and decision making models.  It is built mainly
on top of [sciunit](https://github.com/scidash/sciunit) and
[gym](https://github.com/openai/gym) libraries.  It uses the same
test-model-capability categorization first implemented in sciunit to
run test suites consisting of several tests on a set of models.

## Main features of ldmunit

### Interactive tests
Testing some models should be performed
iteratively. Instead of presenting all the stimuli at once, these can be
inputted one at a time, while observing the actions of the model being
tested. ldmunit formalizes this notion in `InteractiveTest` interface.

### sciunit and gym interaction
In sciunit framework, models are
tagged with capabilities which define the tests a model can possibly
take. ldmunit combines this idea with action and observation spaces
from gym library so that tests can require models to work on for example
continuous action spaces.

### Associative learning model implementations
We provide example
implementations for well-known associative learning models. These models
also demonstrate how to satisfy the interfaces required by interactive
tests that require log probability distributions as predictions. Currently
implemented associative learning models are
* Random respond
* Beta-binomial
* Rescorla-Wagner
* Kalman Rescorla-Wagner
<!-- TODO: what is LSSPD? -->
* LSSPD

### Decision making model implementations
Similarly, we also
provide example implementations for well-known decision making
models. Currently implemented decision making models are
* Random respond
* Rescorla-Wagner Choice Kernel
* Rescorla-Wagner
* Choice Kernel
* Noisy-win-stay-lose-shift

### Support for both single- and multi-subject models
Some models operate on a single-subject at a time (single-subject model) whereas
others can operate on multiple subjects at the same time (multi-subject
model). ldmunit interfaces support multi-subject models by assuming
the model implementation of required interface functions take the
subject index as the first argument. Since a single-subject model can
be represented as a multi-subject model with only one subject, there is
no explicit support for single-subject interfaces. However, we provide a
utility function `ldmunit.models.utils.multi_from_single_interactive` that
takes a single-subject model class as input and returns a multi-subject
model class. Later, this class can be used with ldmunit.

## Installation
You can install ldmunit by running

```bash
pip install ldmunit
```

## Documentation
We provide a series of jupyter notebooks that describe LDMUnit:
<!-- TODO: Below links should point to github
because then they will work from readthedocs and so on. -->
<!-- TODO: change the name of the repository to ldmunit -->
* [Chapter 1: Introduction](docs/ch01_introduction.ipynb)
* [Chapter 2: Tests](docs/ch02_tests.ipynb)
* [Chapter 3: Models](docs/ch03_models.ipynb)
* [Chapter 4: Complete Example](docs/ch04_complete_example.ipynb)

<!-- TODO: API reference link must be https (readthedocs) -->
Additionally, you can browse our <!-- [API reference](TODO) --> to get
more information about certain functions, classes, etc. you want to use.

## Tests
We use built-in `unittest` module for testing ldmunit. To perform checks,
clone this repository and type

```bash
./test.sh
```

## Information for Developers
If you want to extend ldmunit, you need to use the development environment
and `conda`.  To create the `ldmunit` conda environment, use

```bash
conda env create -f environment.yml
conda activate ldmunit
```

We use `black` for code-formatting and `pre-commit` for ensuring high quality code.
To enable these tools simply run

```bash
pre-commit install
```

The next time you try to commit, all the required tools and hooks will be downloaded
(and cached) and checks will be performed on your code.

### Generating Local Documentation
After enabling the development environment, you can generate a local
version of the documentation by running

```bash
cd docs/sphinx
make html
```

Afterwards, you can browse the local documentation by opening
`docs/sphinx/_build/index.html`.

### Installing a Local Development Version
After implementing some changes, you can install the modified version of
ldmunit to your local system by running

```bash
python setup.py install --user
```

Then, every time you import ldmunit, this modified version will be
imported.

## License
<!-- TODO: LICENSE link must be https (github) -->
LDMUnit is distributed under [MIT license](https://opensource.org/licenses/MIT).
See the [LICENSE](LICENSE) file for the exact terms and conditions.
