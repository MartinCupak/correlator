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
    char *inputData, *outputData;
    size_t insize, outsize;
    read_data_from_file(dataRootDir + "/xGPU/input_array_128_128_128_100.bin", inputData, insize);
    read_data_from_file(dataRootDir + "/xGPU/output_matrix_128_128_128_100.bin", outputData, outsize);

    // this one contains arrays / timing configuration etc
    XGPUInfo xgpu_info;
    // this one contains the data itself - need to be allocated!
    XGPUContext xgpu_context;

    // Populate XGPUInfo structure with compile-time parameters
    xgpuInfo(&xgpu_info);

    // re-cast inputData to ComplexInput

    // call xgpuInit function of xGPU library
    xgpu_error = xgpuInit(&xgpu_context, ctx->device);

    // main xGPU call
    xgpu_error = xgpuCudaXengine(&xgpu_context, SYNCOP_SYNC_COMPUTE);
    std::cout << "mwax_db2correlate2db_io_block: CudaXengine done, xgpu_error = " << xgpu_error << std::endl;


    ObservationInfo obsInfo {VCS_OBSERVATION_INFO};
    obsInfo.nTimesteps = 100; // xgpu processes 1/10th of the total 1hour VCS observation at a time.
    auto voltages = Voltages::from_memory((int8_t*) inputData, insize, obsInfo, 100);
    #ifdef __GPU__
    auto xcorr = cross_correlation(voltages);
    xcorr.to_cpu();
    #else
    auto xcorr = cross_correlation_cpu(voltages);
    #endif
    const std::complex<float>* a {reinterpret_cast<std::complex<float>*>(outputData)};
    const std::complex<float>* b {xcorr.data()};
    // xGPU does not compute the time average and does not average channels, so we need to scale back
    // the correlator result.
    const float factor {static_cast<float>(obsInfo.timeResolution * voltages.nIntegrationSteps)};
    for(size_t i {0}; i < xcorr.size(); i++){
        if(a[i] != (b[i] * factor)){
            std::cout << "Elements at position " << i << " differs: " << "a[i] = " << a[i] << ", b[i] = " << b[i] << std::endl;
            throw TestFailed("test_corrrelation_with_xgpu_data failed.");
        }
     }

    xgpuFree(&xgpu_context);

    delete[] inputData;
    delete[] outputData;
    std::cout << "'test_correlation_with_xgpu_data' passed." << std::endl;
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
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Tests batch execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
    } catch (std::exception& ex){
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed." << std::endl;
    return 0;
}
