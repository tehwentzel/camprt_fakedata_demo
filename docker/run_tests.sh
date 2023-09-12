#!/bin/bash

cd ..
cd PYTHON/

jupyter nbconvert --to script "*.ipynb"

#what notebooks to run
python3 SensitivityAnalysis.py

