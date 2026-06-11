#!/usr/bin/bash

echo "Building the Blink correlator for CUDA..."

[ -d build_cpu ] || mkdir build_cpu
cd build_cpu
# note: USE_OPENMP=ON is just for readability, it is ON by default unless set OFF
cmake .. -DBLINK_TEST_DATADIR="${HOME}/data/blink-test-data" \
         -DUSE_OPENMP=ON \
         -DMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3"

make_cpus=$(nproc)
# use 1 .. 8 cpus to compile, and never all - max n-1.
(( make_cpus > 8 )) && make_cpus=8 || make_cpus=$((make_cpus - 1)) && (( make_cpus < 1 )) && make_cpus=1

make -j ${make_cpus} VERBOSE=1
sudo make install
