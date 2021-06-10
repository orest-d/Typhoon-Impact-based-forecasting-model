#!/bin/bash

#########################################################################################################
##  This downloads and installs climada. It should not be necessary since a copy of climada is in lib. ##
#########################################################################################################

wget https://github.com/CLIMADA-project/climada_python/archive/refs/tags/v2.1.1.zip -O climada_python-2.1.1.zip
unzip climada_python-2.1.1.zip
cd climada_python-2.1.1
python setup.py install