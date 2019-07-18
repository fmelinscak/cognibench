#!/usr/bin/env bash

yapf --in-place --recursive --parallel --style='{based_on_style: pep8, indent_width: 4, column_limit: 120}' ldmunit tests docs/*.py
