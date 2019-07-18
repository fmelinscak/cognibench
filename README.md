# LDMUnit
LDMUnit is a framework for test and validation of learning and decision making models.
It is built mainly on top of [sciunit](https://github.com/scidash/sciunit) and [gym](https://github.com/openai/gym) libraries.
It uses the same test-model-capability categorization first implemented in sciunit to run test suites
consisting of several tests on a set of models.

## Installation
You can install ldmunit by running

```bash
pip install ldmunit
```

## Documentation
We provide a series of jupyter notebooks that describe LDMUnit:
<!-- TODO: Below links should point to github because then they will work from readthedocs and so on. -->
<!-- TODO: change the name of the repository to ldmunit -->
* [Chapter 1: Introduction](docs/ch01_introduction.ipynb)
* [Chapter 2: Tests](docs/ch02_tests.ipynb)
* [Chapter 3: Models](docs/ch03_models.ipynb)
* [Chapter 4: Complete Example](docs/ch04_complete_example.ipynb)

<!-- TODO: API reference link must be https (readthedocs) -->
Additionally, you can browse our <!-- [API reference](TODO) --> to get more information about certain
functions, classes, etc. you want to use.

## Tests
We use built-in `unittest` module for testing ldmunit. To perform checks, type

```bash
./test.sh
```

## Information for Developers
If you want to extend ldmunit, you need to use the development environment and `conda`.
To create the `ldmunit_dev` conda environment, use

```bash
conda env create -f environment_dev.yml
conda activate ldmunit_dev
```

### Generating Local Documentation
After enabling the development environment, you can generate a local version of the documentation
by running

```bash
cd docs/sphinx
make html
```

Afterwards, you can browse the local documentation by opening `docs/sphinx/_build/index.html`.

### Installing a Local Development Version
After implement some changes, you can install the modified version of ldmunit to your
local system by running

```bash
python setup.py install --user
```

Then, every time you import ldmunit, this modified version will be imported.

## License
<!-- TODO: LICENSE link must be https (github) -->
LDMUnit is distributed under [MIT license](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file
for the exact terms and conditions.
