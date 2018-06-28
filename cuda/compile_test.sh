#!/bin/sh
#nvcc -m 32 -arch sm_35 -Xptxas=-v,-abi=no -cubin sha256.cu -D_FORCE_INLINES
#nvcc -arch sm_35 main.cu sha256.cu -D_FORCE_INLINES -lcudart -o main

#nvcc -Xptxas -dlcm=ca -Xptxas -dscm=cs -gencode=arch=compute_35,code=\"sm_35,compute_35\"  main.cu sha256.cu -D_FORCE_INLINES -lcudart -o main
#nvcc -g -O2 -gencode=arch=compute_35,code=\"sm_35,compute_35\"  main.cu sha256.cu -D_FORCE_INLINES -lcudart -o main

nvcc -Xptxas -dlcm=ca -Xptxas -dscm=cs -gencode=arch=compute_35,code=\"sm_35,compute_35\"  main_test.cu haraka.cu -D_FORCE_INLINES -lcudart -o main_test
./main_test
#nvcc -Xptxas -dlcm=ca -Xptxas -dscm=cs -gencode=arch=compute_35,code=\"sm_35,compute_35\"  main.cu haraka.cu -D_FORCE_INLINES -lcudart -o main

#export GPU_FORCE_64BIT_PTR=0
#export GPU_MAX_HEAP_SIZE=100
#export GPU_USE_SYNC_OBJECTS=1
#export GPU_MAX_ALLOC_PERCENT=100

#gnome-terminal -e "time ./main"
