#include <exception>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <stdexcept>

// xGPU includes
#include "xgpu.h"

#include <astroio.hpp>
#include "../src/utils.hpp"
#include "../src/correlation.hpp"
#include "../src/correlation.h"
#include "common.hpp"

#include <cstdio>

#include <malloc.h>

std::string dataRootDir;


template <typename T>
bool complex_vectors_equal(const std::complex<T>* a, const std::complex<T>* b, size_t length){
    double delta;
    const double TOL {0};
    for(size_t i {0}; i < length; i++){
        if (std::abs(a[i]) == 0) 
            delta = std::abs(b[i]);
        else 
            delta = std::abs(a[i] - b[i]);
        
        if (delta > TOL) {
            std::cout << "Elements at position " << i << " differs (delta = " << delta <<"): " << "a[i] = " << a[i] << ", b[i] = " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}


void test_xgpu_correlation(){
#ifndef __GPU__
    std::cout << "Sorry mate, cannot run 'test_xgpu_correlation()', xGPU kinda needs GPU :-(" << std::endl;
#else

    int xgpu_error = 0;
#define DEFAULT_DEVICE_ID 0
    int device = (int)DEFAULT_DEVICE_ID;

    // this one contains arrays / timing configuration etc
    XGPUInfo xgpu_info;
    // this one contains the data itself - need to be allocated!
    XGPUContext xgpu_context;

    // Populate XGPUInfo structure with compile-time parameters
    // = Get telescope (antenna arrays) sizing & configuration info from xGPU library
    // (function has void return, so can't error check)
    xgpuInfo(&xgpu_info);
    std::cout << "test_xgpu_correlation(): xGPU configured for "
            << "NSTATION aka nAntennas = " << xgpu_info.nstation << ", "
            << "NPOL aka nPolarizations = " << xgpu_info.npol << ", "
            << "NTIME aka nTimesteps = " << xgpu_info.ntime << ", "
            << "NTIME_PIPE = " << xgpu_info.ntimepipe << ", "
            << "and output matrix order = " << xgpu_info.matrix_order
            << std::endl;

    // MCu note: fill up other perams? or not needed?
    /*  ===> Nope, redundant! That is a part of the telescope configuration reflected in xGPU build
    xgpu_info.ntime = 100;
    // ??? xgpu_info.ntimepipe = 100;
    */

    // setting the following to NULLs tells xgpuInit() to allocate the buffers, pin it, and set the pointers
    xgpu_context.array_h  = NULL;
    xgpu_context.matrix_h = NULL;
    // typically only a small fraction of matrix_h will get populated - beginning at the start address
    // call xgpuInit function of xGPU library
    xgpu_error = xgpuInit(&xgpu_context, device);

    // MCu note: this is defined in xGPU::xgpu.h
    // legacu xGPU uses integer input data, mwax xGPU float
#ifndef FIXED_POINT
    std::cout << "test_xgpu_correlation(): Sending floating point data to GPU." << std::endl;
#else
    std::cout << "test_xgpu_correlation(): Sending fixed point data to GPU." << std::endl;
#endif

    // read data using function from Blink correlator::utils.xpp into char *inputData, *outputData;
    char *inputData, *outputData;
    size_t insize, outsize;
    read_data_from_file(dataRootDir + "/xGPU/input_array_128_128_128_100.bin", inputData, insize);
    read_data_from_file(dataRootDir + "/xGPU/output_matrix_128_128_128_100.bin", outputData, outsize);

/*
    // get inputData with astrolib::Voltages class (nice C++)
    // ObservationInfo type struct needed to initialise Voltage class
    ObservationInfo obsInfo {VCS_OBSERVATION_INFO};
    obsInfo.nTimesteps = xgpu_info.ntime;
    // timestepsPerRead=100 ... is that the same thing as obsInfo.nTimesteps? - No, this is nIntegrationSteps
    // TODO: ask Cristian
    auto voltages = Voltages::from_memory((int8_t*) inputData, insize, obsInfo, 100);
*/
    // re-cast inputData to ComplexInput
    //const std::complex<int8_t>* in {reinterpret_cast<std::complex<int8_t>*>(inputData)};
    std::cout << "size of inputData from file = " << insize << std::endl;
    std::cout << "malloc_usable_size of inputData from file = " << malloc_usable_size(inputData) << std::endl;
    std::cout << "this is funny, malloc_usable_size() !" << std::endl;
    // segfault std::cout << "malloc_usable_size of xgpu_context.array_h = " << malloc_usable_size(xgpu_context.array_h) << std::endl;
    xgpu_context.array_h = reinterpret_cast<ComplexInput*>(inputData);

    // main xGPU call
    xgpu_error = xgpuCudaXengine(&xgpu_context, SYNCOP_SYNC_COMPUTE);
    std::cout << "test_xgpu_correlation(): CudaXengine done, xgpu_error = " << xgpu_error << std::endl;

/*  blink correlator test code
    ObservationInfo obsInfo {VCS_OBSERVATION_INFO};
    obsInfo.nTimesteps = 100; // xgpu processes 1/10th of the total 1hour VCS observation at a time.
    # nIntegrationSteps
    auto voltages = Voltages::from_memory((int8_t*) inputData, insize, obsInfo, 100);
    auto xcorr = cross_correlation(voltages);
    xcorr.to_cpu();
*/
    const std::complex<float>* a {reinterpret_cast<std::complex<float>*>(outputData)};
    // Complex* b {reinterpret_cast<Complex*>(xgpu_context.matrix_h)};
    Complex* b = (Complex*)xgpu_context.matrix_h;
    // const std::complex<float>* b {xcorr.data()};
    // xGPU does not compute the time average and does not average channels, so we need to scale back
    // the correlator result.
    // const float factor {static_cast<float>(obsInfo.timeResolution * voltages.nIntegrationSteps)};
    const unsigned int nIntegrationSteps = xgpu_info.matrix_order;
    const float factor {static_cast<float>(xgpu_info.ntime * nIntegrationSteps)};
    for(size_t i {0}; i < outsize/sizeof(Complex); i++){
        if(a[i] != (b[i] * factor)){
            std::cout << "Elements at position " << i << " differs: " << "a[i] = " << a[i] << ", b[i] = " << b[i] << std::endl;
            throw TestFailed("test_corrrelation_with_xgpu_data failed.");
        }
     }

    xgpuFree(&xgpu_context);

    delete[] inputData;
    delete[] outputData;
    std::cout << "'test_correlation_with_xgpu_data' passed." << std::endl;
#endif
}


int main(void){
    char *pathToData {std::getenv(ENV_DATA_ROOT_DIR)};
    if(!pathToData){
        std::cerr << "'" << ENV_DATA_ROOT_DIR << "' environment variable is not set." << std::endl;
        return -1;
    }
    dataRootDir = std::string {pathToData};

    try{
        auto start = std::chrono::high_resolution_clock::now();
        //test_correlation_with_xgpu_data();
        test_xgpu_correlation();
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Tests batch execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
    } catch (std::exception& ex){
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed." << std::endl;
    return 0;
}
