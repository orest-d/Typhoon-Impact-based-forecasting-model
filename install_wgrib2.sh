#!/bin/bash

### May require Fortran 95
# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt update
# sudo apt install gfortran-9

wget https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz.v3.0.2 -O wgrib2.tgz
tar -xvzf wgrib2.tgz

cd grib2
export CC=gcc
export FC=gfortran
make
cd ..


cp grib2/wgrib2/wgrib2 wgrib2