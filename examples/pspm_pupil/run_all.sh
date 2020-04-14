#!/usr/bin/env bash

for i in {1..7}; do
    echo "RUNNING EXPERIMENT ${i}..."
    python exp${i}/bench.py
done
