#!/usr/bin/bash

nvcc correlation_test.cpp -o correlation_test  -I/usr/local/include -I/usr/local/cuda-11.8/include -L/usr/local/lib -L/usr/local/cuda-11.8/lib64 -lpsrdada -lcudart -lm -lpthread -lblink_astroio -lcorrelation
