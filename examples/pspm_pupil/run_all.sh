#!/usr/bin/env bash
#
# STEPS
# 1. If there is no data folder, download datasets using download_datasets.sh
# 2. Change PsPM src folder path in libcommon/util.py line 14
# 3. Run this script
#
# Outputs will be stored in output folder
# Logs will be stored in log folder

mkdir -p log
for i in {1..7}; do
    echo "RUNNING EXPERIMENT ${i}..."
    mkdir -p log/exp${i}
    python exp${i}/bench.py > log/exp${i}/stdout 2> log/exp${i}/stderr &
done
