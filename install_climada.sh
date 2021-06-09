#!/bin/bash

wget https://github.com/CLIMADA-project/climada_python/archive/refs/tags/v2.1.1.zip -O climada_python-2.1.1.zip
unzip climada_python-2.1.1.zip
cd climada_python-2.1.1
python setup.py install