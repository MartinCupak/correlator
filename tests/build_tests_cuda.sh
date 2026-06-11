#!/usr/bin/bash

# must be full absolute path, will not work with ~/.../...
export BLINK_TEST_DATADIR="${HOME}/data/blink-test-data"

nvcc correlation_test.cpp -o correlation_test -O3 -I/usr/local/include -I/usr/local/cuda-11.8/include -L/usr/local/lib -L/usr/local/cuda-11.8/lib64 -lpsrdada -lcudart -lm -lpthread -lblink_astroio -lcorrelation
./correlation_test

# nvcc xgpu_test.cpp -o xgpu_test -O3 -I/usr/local/include -I/usr/local/cuda-11.8/include -L/usr/local/lib -L/usr/local/cuda-11.8/lib64 -lpsrdada -lcudart -lm -lpthread -lblink_astroio -lcorrelation -lxgpu_128_128_128_100
