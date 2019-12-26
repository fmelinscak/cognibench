# Description
In this example we test three hypothetical submissions made to CPC18 track I. All three models are slight variations of
the baseline BEASTsd model provided by CPC18 committee. We assume that every participant submitted a folder containing
their models and the main script provided by the CPC committee.

For showcasing purposes, each of the three models are implemented in different languages (Python, Octave and R).
`ldmunit` can easily test models implemented in these languages with the use of wrappers we provide as long as model
source code structure is as specified by the respective wrapper class. You can see the model definitions required to
use these models in `model_defs.py` file.

# Running
Simply type
```bash
python test_cpc_contestants.py
```

# Acknowledgements
Model implementations and data in this folder are obtained from CPC18 official website:
https://cpc-18.com/baseline-models-and-source-code
