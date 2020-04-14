#!/usr/bin/env bash

matlabpath="/usr/local/MATLAB/R2019b"
orig_path=`pwd`
cd ${matlabpath}/extern/engines
sudo chown -R ubuntu:ubuntu python
cd python
python setup.py install
cd ${orig_path}
