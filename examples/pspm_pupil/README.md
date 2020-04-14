# PsPM Pupil Benchmarking
This folder contains the benchmarking codes for seven different experiments that measure the retrodictive validity
of various psychophysiological models. In each experiment, a series of models that are created by combining certain
configuration parameters are tested against seven different pupil datasets. The datasets, along with their links, are
given below:

## Datasets
1. DoxMeM2 (only placebo group): https://doi.org/10.5281/zenodo.3441715
2. FER02: http://doi.org/10.5281/zenodo.3555306
3. FSS6B: https://doi.org/10.5281/zenodo.3601250
4. LI: https://doi.org/10.5281/zenodo.1288493
5. PubFe: https://doi.org/10.5281/zenodo.1168493
6. SC4B: https://doi.org/10.5281/zenodo.1039580
7. VC7B: https://doi.org/10.5281/zenodo.1211609


## Running
To run a specific experiment simply type, e.g. experiment 1,
```bash
python exp1/bench.py
```

To run the whole pipeline in parallel (careful, requires **lots** of memory since we spawn 7 separate MATLAB processes)
```bash
./run_all.sh
```

## Project Structure (for Developers)
This section explains the structuring of the whole benchmarking pipeline, how python and MATLAB interacts,
how models are implemented and so on.

### Overall Structure
Each experiment is given its own folder (`exp1`, `exp2#`, etc.). In each of these folders, we have two core files:

1. `bench.py`: This is the driver python script that integrates Cognibench, model scripts and PsPM. This script
defines the overall experiment parameters and creates models for each combination. Afterwards, the whole test suite
is run in the usual cognibench way.

2. `fit.m`: Each experiment has this MATLAB function that defines the fitting procedure of a single subject. Whether GLM
or DCM is used, the type of preprocessing steps should all go to this file.

3. `model_defs.py`: This python file defines the PsPM model wrapper. In the current version, a PsPM model is
defined as one that fits all the subject data and returns the parameter fits of all subjects. However, this idea
is not yet realized with cognibench multi subject model <!--(TODO)-->

### Common Functions
Many of the common functionalities that are used by all experiments are defined in `libcommon`.
This folder contains MATLAB functions for

1. Reading condition files (each dataset has slightly different conventions or requirements).
2. Common MATLAB utility functions
3. `fit_all.m`: This is the MATLAB function that uses an experiment's `fit` function to fit all the subjects in a dataset.
It adds error handling and dataset copying/removing functionalities on top of a simple for loop.
4. Pupil preprocessing utility functions (`pp_pfe.m` and `pp_valid_fixations.m`)
5. `util.py`: Common python utilities and class definitions

### Scripts
1. `download_datasets.sh`: This is the script to download all datasets from zenodo, unzip them and remove SCR
or other unnecessary files
2. `install_matlab_engine.sh`: This is specifically for ScienceCloud, and is only required if a new instance is set up.
3. `run_all.sh`: A simple do-it-all script to run the whole pipeline. Experiments run in parallel by spawning a separate
MATLAB process for each experiment.
