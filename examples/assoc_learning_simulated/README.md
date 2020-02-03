# Testing of Associative Learning Models
This example folder provides a sample implementation of a testing script using `cognibench` library. The script is
implemented in `run_all_tests.py`. It runs three test suites: each suite runs an interactive testing of associative
learning models against previously generated data using `NLL`, `AIC` or `BIC` scores. To run the script, simply type

```
python run_all_tests.py
```

This should persist the test results under `results` directory. Each folder in this directory belongs to one of the
test cases.
