#/usr/bin/bash

export BLINK_TEST_DATADIR="/home/mcu/data/blink-test-data"
# export CMAKE_CUDA_ARCHITECTURES="61-real"
export USE_HIP=ON
export CMAKE_C_COMPILER=hipcc
export CMAKE_CXX_COMPILER=hipcc
export CMAKE_CXX_FLAGS="--offload-arch=gfx90a -O3"
export CMAKE_BUILD_TYPE=Release
