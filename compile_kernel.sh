#!/bin/sh
curdir=$(pwd)
echo Curdir: $curdir
cd $curdir/cuda
./compile_kernel.sh
cd $curdir
