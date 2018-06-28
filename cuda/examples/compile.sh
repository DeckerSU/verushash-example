#!/bin/sh
nvcc -Xptxas -dlcm=ca -Xptxas -dscm=cs -gencode=arch=compute_35,code=\"sm_35,compute_35\" $1 -D_FORCE_INLINES -lcudart -o $1.bin
./$1.bin
