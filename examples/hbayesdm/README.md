# Testing hBayesDM Models
This folder implements a testing script to test models implemented using python version of `hBayesDM` library.

To test such models a user needs to write a wrapper model in `cognibench` to delegate the function calls to corresponding
calls in `hBayesDM`. This is provided in `model_defs.py` file. Here, `HbayesdmModel` class implements the `fit` and
`predict` methods.

`run_tests.py` is the main testing script. It defines several models using our `hBayesDM` wrapper class and tests them
against three batch test classes, one for each of `NLL`, `AIC` and `BIC` scoring.

## Disclaimer
`HbayesdmModel` class we provide requires the model implementations from hBayesDM (and in turn STAN) to return the
probability distribution over the action space rather than just a single prediction. To facilitate this, we have
slightly modified the `hBayesDM` model implementations we use in this example. These are accessible at https://github.com/eozd/hbayesdm
where you can see our small changes. If you would like to use `hBayesDM`, then please adjust the STAN model implementations
to return a probability distribution over the action space so that you can integrate your models into `cognibench` easily.
