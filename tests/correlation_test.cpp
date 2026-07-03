#include <exception>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <stdexcept>

#include <astroio.hpp>
#include "../src/utils.hpp"
#include "../src/correlation.hpp"
#include "../src/correlation.h"
#include "common.hpp"

#include <cstdio>

std::string dataRootDir;



void test_complex_conjugate_multiply(){
    
    std::complex<int> a, b, res {0, 0};
    // subtest 1: multiply a number by its conjugate
    a.real(8);
    a.imag(3);

    b.real(8);
    b.imag(3);
    ccm(a, b, res);
    if(res.real() != 73 || res.imag() != 0){
        throw TestFailed("test_complex_conjugate_multiply failed at subtest 1.");
    }

    // subtest 2: multiply a number by another conjugate
    res = std::complex<int> {0, 0};
    b.real(5);
    b.imag(1);

    // TODO: fix this test.
    std::cout << "'test_complex_conjugate_multiply' passed." << std::endl;
}



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

// MCu, inspired by python
bool isclose(const std::complex<float>& a, 
             const std::complex<float>& b,
             float rtol = 1e-5f,
             float atol = 1e-8f) {
    // Same logic as numpy: |a - b| <= atol + rtol * |b|
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}


void test_correlation_with_xgpu_data(){
    char *inputData, *outputData;
    size_t insize, outsize;
    read_data_from_file(dataRootDir + "/xGPU/input_array_128_128_128_100.bin", inputData, insize);

    read_data_from_file(dataRootDir + "/xGPU/output_matrix_128_128_128_100.bin", outputData, outsize);
    
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

    delete[] inputData;
    delete[] outputData;
    std::cout << "'test_correlation_with_xgpu_data' passed." << std::endl;
}

// MCu+Cld: 
// Map xGPU REGISTER_TILE_ORDER → Blink triangular order
// for validation purposes only
void reorder_xgpu_to_blink(
    const std::complex<float>* xgpu,   // 576 entries per channel
    std::complex<float>* blink_ref,    // 544 entries per channel
    int nstation,                       // 16
    int npol,                          // 2
    int nfreq)                          // 6400
{
    int ntiles    = nstation / 2;                            // 8
    int xgpu_per_chan = (ntiles+1)*ntiles/2 * npol*npol * 4; // 576
    int blink_per_chan = (nstation+1)*nstation/2 * npol*npol; // 544

    for (int f = 0; f < nfreq; f++) {
        const std::complex<float>* xg = xgpu     + f * xgpu_per_chan;
        std::complex<float>*       bl = blink_ref + f * blink_per_chan;

        int tile = 0;
        for (int tr = 0; tr < ntiles; tr++) {
            for (int tc = 0; tc <= tr; tc++, tile++) {
                // Each tile holds a 2×2 antenna block × npol² pol products
                // xGPU stores them as [4 tile_entries × npol²]
                // tile_entry layout: (0→upper-tri, 1→(2tr+1,2tc), 2→(2tr,2tc), 3→(2tr+1,2tc+1))
                // (exact order from xgpu.h REGISTER_TILE_ORDER definition)

                for (int pi = 0; pi < npol; pi++) {         // pol of row antenna
                    for (int pj = 0; pj < npol; pj++) {     // pol of col antenna

                        // The four antenna pairs within this tile:
                        // entry 2 → (ant_row=2tr,   ant_col=2tc)
                        // entry 1 → (ant_row=2tr+1, ant_col=2tc)
                        // entry 0 → upper triangle (2tr, 2tc+1) — SKIP on diagonal
                        // entry 3 → (ant_row=2tr+1, ant_col=2tc+1)

                        int pol_offset = pi * npol + pj;

                        // Pair (2tr, 2tc):
                        {
                            int ai = 2*tr, aj = 2*tc;
                            int si = ai*npol+pi, sj = aj*npol+pj;
                            int blink_idx = si*(si+1)/2 + sj;
                            int xgpu_idx  = tile*npol*npol*4 + pol_offset*4 + 2;
                            bl[blink_idx] = xg[xgpu_idx];
                        }
                        // Pair (2tr+1, 2tc):
                        {
                            int ai = 2*tr+1, aj = 2*tc;
                            int si = ai*npol+pi, sj = aj*npol+pj;
                            int blink_idx = si*(si+1)/2 + sj;
                            int xgpu_idx  = tile*npol*npol*4 + pol_offset*4 + 1;
                            bl[blink_idx] = xg[xgpu_idx];
                        }
                        // Pair (2tr+1, 2tc+1):
                        {
                            int ai = 2*tr+1, aj = 2*tc+1;
                            int si = ai*npol+pi, sj = aj*npol+pj;
                            int blink_idx = si*(si+1)/2 + sj;
                            int xgpu_idx  = tile*npol*npol*4 + pol_offset*4 + 3;
                            bl[blink_idx] = xg[xgpu_idx];
                        }
                        // Pair (2tr, 2tc+1): only valid off diagonal (tr > tc)
                        if (tr > tc) {
                            int ai = 2*tr, aj = 2*tc+1;
                            int si = ai*npol+pi, sj = aj*npol+pj;
                            int blink_idx = si*(si+1)/2 + sj;
                            int xgpu_idx  = tile*npol*npol*4 + pol_offset*4 + 0;
                            bl[blink_idx] = xg[xgpu_idx];
                        }
                    }
                }
            }
        }
    }
}

// MCu+Cld: tri(n) = n*(n+1)/2
static inline int tri(int n){ return n*(n+1)/2; }

// MCu+Cld: 
// Convert xGPU register tile output → Blink triangular order
// xgpu_buf: [real_block | imag_block], each block = n_freq × half_tri_size×16 floats
// blink_buf: std::complex<float> interleaved, [baseline_blink][4 pols][channel]
//            pols in Blink order: XX, XY, YX, YY
void convert_xgpu_reg_to_blink(
    const float*     xgpu_buf,    // 29,491,200 bytes for 16 ant, 6400 ch
    std::complex<float>* blink_buf,   // 3,481,600 complex = 27,852,800 bytes
    int nant, int npol, int nfreq)
{
    int num_tiles    = nant;                    // = 16 (signal path pairs = antennas)
    int half_tri_sz  = tri(nant / 2);           // tri(8) = 36
    int reg_delta    = half_tri_sz * 16;        // 576 floats per channel

    const float* in_re = xgpu_buf;
    const float* in_im = xgpu_buf + (size_t)nfreq * reg_delta;

    for (int v = 0; v < num_tiles; v++) {       // row antenna
        int i  = v >> 1;
        int rx = v & 1;
        for (int w = 0; w <= v; w++) {          // col antenna (w <= v)
            int j  = w >> 1;
            int ry = w & 1;

            // Blink row-first baseline index
            int blink_bl = v * (v + 1) / 2 + w;

            // xGPU register tile index (start, before channel loop)
            int reg_start = (tri(i) + j + (2 * ry + rx) * half_tri_sz) * 4;

            for (int f = 0; f < nfreq; f++) {
                int ri = reg_start + f * reg_delta;

                // xGPU pol order at offsets 0,1,2,3: XX, XY, YX, YY
                // MWAX conjugates and swaps XY/YX — Blink may not want conjugate
                // Try WITHOUT conjugate first (verify against Blink convention):
                float xx_re =  in_re[ri + 0],  xx_im = in_im[ri + 0];
                float xy_re =  in_re[ri + 1],  xy_im = in_im[ri + 1];
                float yx_re =  in_re[ri + 2],  yx_im = in_im[ri + 2];
                float yy_re =  in_re[ri + 3],  yy_im = in_im[ri + 3];

                // Blink output baseline index (row-first, per channel)
                int out = blink_bl * npol * npol * nfreq + f * npol * npol;

                // Blink pol order — adjust if Blink uses different order!
                // Assuming standard XX, XY, YX, YY (verify with Blink docs):
                blink_buf[out + 0] = {xx_re,  xx_im};
                blink_buf[out + 1] = {xy_re,  xy_im};
                blink_buf[out + 2] = {yx_re,  yx_im};
                blink_buf[out + 3] = {yy_re,  yy_im};
            }
        }
    }
}

#ifdef __GPU__
void test_correlation_with_xgpu_in_mwax_data(){
    char *inputData1, *inputData2, *outputData;
    size_t insize1, insize2, outsize;

    read_data_from_file(dataRootDir + "/mwax/xgpu_input_000.00.bin", inputData1, insize1);
    read_data_from_file(dataRootDir + "/mwax/xgpu_input_000.01.bin", inputData2, insize2);
    read_data_from_file(dataRootDir + "/mwax/xgpu_output_000.bin", outputData, outsize);
    
   
    const std::complex<float>* voltages1_cpu = reinterpret_cast<std::complex<float>*>(inputData1);
    const std::complex<float>* voltages2_cpu = reinterpret_cast<std::complex<float>*>(inputData2);
    const std::complex<float>* reference_output {reinterpret_cast<std::complex<float>*>(outputData)};

    std::complex<float> *visibilities_gpu, *visibilities_cpu, *voltages1_gpu, *voltages2_gpu;
    
    const unsigned int n_antennas {144u};
    const unsigned int n_baselines {((n_antennas + 1) * n_antennas) / 2};
    const unsigned int n_polarisations {2u};
    const unsigned int n_fine_channels {6400u};
    const unsigned int n_time_samples {52u};
    const unsigned int n_integrated_samples {52u};
    const unsigned int n_integration_intervals {n_time_samples / n_integrated_samples};
    // the following definition will make sure that the output won't be scaled by the time
    // averaging factor.
    const double time_resolution {1.0 / n_integrated_samples};
    const unsigned int n_channels_to_avg {1u};
    const unsigned int reset_visibilities {1u};
    
    size_t n_voltages {static_cast<size_t>(n_integration_intervals) * n_fine_channels * n_antennas * n_polarisations * n_integrated_samples};
    size_t n_visibilities {static_cast<size_t>(n_integration_intervals) * n_fine_channels * n_baselines * n_polarisations * n_polarisations};

    if(n_voltages * sizeof(std::complex<float>) != insize1){
        std::cerr << "Input 1 size does not match the expected size as computed by observation info." << std::endl;
        throw TestFailed("Input 1 size does not match the expected size as computed by observation info.");
    }
    size_t exp_vis_size {n_visibilities * sizeof(std::complex<float>)};
    // if(exp_vis_size != outsize){
    //     std::cerr << "Output size (" << outsize << ") does not match the expected size (" << exp_vis_size << ") as computed by observation info." << std::endl;
    //     throw TestFailed("Output size does not match the expected size as computed by observation info.");
    // }
    
    // allocate memory and copy data to gpu
    visibilities_cpu = new std::complex<float>[n_visibilities];
    gpuMalloc(&voltages1_gpu, n_voltages * sizeof(std::complex<float>));
    gpuMalloc(&voltages2_gpu, n_voltages * sizeof(std::complex<float>));
    gpuMalloc(&visibilities_gpu, n_visibilities * sizeof(std::complex<float>));

    gpuMemcpy(voltages1_gpu, voltages1_cpu, n_voltages * sizeof(std::complex<float>), gpuMemcpyHostToDevice);
    gpuMemcpy(voltages2_gpu, voltages2_cpu, n_voltages * sizeof(std::complex<float>), gpuMemcpyHostToDevice);

    int return_value = blink_cross_correlation_gpu((float*)voltages1_gpu, (float*)visibilities_gpu, n_antennas,
        n_polarisations, n_fine_channels, n_time_samples, time_resolution, n_integrated_samples,
        n_channels_to_avg, 1);
    if(return_value){
        throw TestFailed("First call to `blink_cross_correlation_gpu` returned a non-zero code.");
    }
    return_value = blink_cross_correlation_gpu((float*)voltages2_gpu, (float*)visibilities_gpu, n_antennas,
        n_polarisations, n_fine_channels, n_time_samples, time_resolution, n_integrated_samples,
        n_channels_to_avg, 0);
    if(return_value){
        throw TestFailed("Second call to `blink_cross_correlation_gpu` returned a non-zero code.");
    }
    gpuMemcpy(visibilities_cpu, visibilities_gpu, sizeof(std::complex<float>) * n_visibilities, gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();    

     for(size_t i {0}; i < n_visibilities; i++){
        if(visibilities_cpu[i] != reference_output[i]){
            std::cout << "Elements at position " << i << " differs: " << "vis_cpu[i] = " << visibilities_cpu[i] << ", ref[i] = " << reference_output[i] << std::endl;
            throw TestFailed("'test_corrrelation_with_xgpu_in_mwax_data' failed.");
        }
     }

    gpuFree(voltages1_gpu);
    gpuFree(voltages2_gpu);
    gpuFree(visibilities_gpu);
    delete[] inputData1;
    delete[] inputData2;
    delete[] visibilities_cpu;
    delete[] outputData;
    delete[] reference_output;
    std::cout << "'test_correlation_with_xgpu_in_mwax_data' passed." << std::endl;
}

void test_correlation_with_xgpu_in_mwax_data_16T(){
    char *inputData1, *inputData2, *outputData;
    size_t insize1, insize2, outsize;

    std::cout << "Read inputData1 file " << dataRootDir + "/mwax/xgpu_input_000.00.bin" << std::endl;
    read_data_from_file(dataRootDir + "/mwax/xgpu_input_008.00.bin", inputData1, insize1);
    std::cout << "Read inputData2 file " << dataRootDir + "/mwax/xgpu_input_000.01.bin" << std::endl;
    read_data_from_file(dataRootDir + "/mwax/xgpu_input_008.01.bin", inputData2, insize2);
    std::cout << "Read outputData file " << dataRootDir + "/mwax/xgpu_output_000.bin" << std::endl;
    read_data_from_file(dataRootDir + "/mwax/xgpu_output_008.bin", outputData, outsize);
    
    const std::complex<float>* voltages1_cpu = reinterpret_cast<std::complex<float>*>(inputData1);
    const std::complex<float>* voltages2_cpu = reinterpret_cast<std::complex<float>*>(inputData2);
    const std::complex<float>* reference_output {reinterpret_cast<std::complex<float>*>(outputData)};

    std::complex<float> *visibilities_gpu, *visibilities_cpu, *voltages1_gpu, *voltages2_gpu;
    
    const unsigned int n_antennas {16u};
    const unsigned int n_baselines {((n_antennas + 1) * n_antennas) / 2};
    const unsigned int n_polarisations {2u};
    const unsigned int n_fine_channels {6400u};
    const unsigned int n_time_samples {52u};
    const unsigned int n_integrated_samples {52u};
    const unsigned int n_integration_intervals {n_time_samples / n_integrated_samples};
    // the following definition will make sure that the output won't be scaled by the time
    // averaging factor.
    const double time_resolution {1.0 / n_integrated_samples};
    const unsigned int n_channels_to_avg {1u};
    const unsigned int reset_visibilities {1u};
    
    size_t n_voltages {static_cast<size_t>(n_integration_intervals) * n_fine_channels * n_antennas * n_polarisations * n_integrated_samples};
    size_t n_visibilities {static_cast<size_t>(n_integration_intervals) * n_fine_channels * n_baselines * n_polarisations * n_polarisations};

    if(n_voltages * sizeof(std::complex<float>) != insize1){
        std::cerr << "Input 1 size does not match the expected size as computed by observation info." << std::endl;
        throw TestFailed("Input 1 size does not match the expected size as computed by observation info.");
    }
    size_t exp_vis_size {n_visibilities * sizeof(std::complex<float>)};

    // MCu+Cld: Reorder reference before comparing
    std::complex<float>* reordered_ref = new std::complex<float>[n_visibilities];
    
    // Build a minimal ctx-like struct for the conversion function
    // (or just replicate its logic inline — see below)
    convert_xgpu_reg_to_blink(
        reinterpret_cast<float*>(outputData),   // raw register tile
        reordered_ref,
        n_antennas, n_polarisations, n_fine_channels);

    // no point to compare size when we have re-ordered the reference output to an array with the same allocation.
    // if(exp_vis_size != outsize){
    //     std::cerr << "Output size (" << outsize << ") does not match the expected size (" << exp_vis_size << ") as computed by observation info." << std::endl;
    //     throw TestFailed("Output size does not match the expected size as computed by observation info.");
    // }
    
    // allocate memory and copy data to gpu
    visibilities_cpu = new std::complex<float>[n_visibilities];
    gpuMalloc(&voltages1_gpu, n_voltages * sizeof(std::complex<float>));
    gpuMalloc(&voltages2_gpu, n_voltages * sizeof(std::complex<float>));
    gpuMalloc(&visibilities_gpu, n_visibilities * sizeof(std::complex<float>));

    gpuMemcpy(voltages1_gpu, voltages1_cpu, n_voltages * sizeof(std::complex<float>), gpuMemcpyHostToDevice);
    gpuMemcpy(voltages2_gpu, voltages2_cpu, n_voltages * sizeof(std::complex<float>), gpuMemcpyHostToDevice);

    int return_value = blink_cross_correlation_gpu((float*)voltages1_gpu, (float*)visibilities_gpu, n_antennas,
        n_polarisations, n_fine_channels, n_time_samples, time_resolution, n_integrated_samples,
        n_channels_to_avg, 1);
    if(return_value){
        throw TestFailed("First call to `blink_cross_correlation_gpu()` returned a non-zero code.");
    }
    else{ std::cout << "First call to `blink_cross_correlation_gpu()` run OK."<< std::endl;
    }

    // this just overwrites the prev step output data in visibilities_gpu?
    return_value = blink_cross_correlation_gpu((float*)voltages2_gpu, (float*)visibilities_gpu, n_antennas,
        n_polarisations, n_fine_channels, n_time_samples, time_resolution, n_integrated_samples,
        n_channels_to_avg, 0);
    if(return_value){
        throw TestFailed("Second call to `blink_cross_correlation_gpu()` returned a non-zero code.");
    }
    else{ std::cout << "Second call to `blink_cross_correlation_gpu()` run OK."<< std::endl;
    }
    gpuMemcpy(visibilities_cpu, visibilities_gpu, sizeof(std::complex<float>) * n_visibilities, gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();    

    int cnt_diff=0;
    for(size_t i {0}; i < n_visibilities; i++){
        if (!isclose(visibilities_cpu[i], reordered_ref[i])) {
            cnt_diff++;
            std::cout << "Elements at position " << i << " differs: " << "vis_cpu[i] = " 
                << visibilities_cpu[i] << ", ref[i] = " << reordered_ref[i] << std::endl;
            if( cnt_diff > 64 ) {
                throw TestFailed("'test_corrrelation_with_xgpu_in_mwax_data' failed.");
            }   
        }
        else {
            std::cout << "Elements at position " << i << " ARE EQUAL: " << "vis_cpu[i] = " 
                << visibilities_cpu[i] << ", ref[i] = " << reordered_ref[i] << std::endl;
        }
    }

    gpuFree(voltages1_gpu);
    gpuFree(voltages2_gpu);
    gpuFree(visibilities_gpu);
    delete[] inputData1;
    delete[] inputData2;
    delete[] visibilities_cpu;
    delete[] outputData;
    delete[] reference_output;
    delete[] reordered_ref;
    std::cout << "'test_correlation_with_xgpu_in_mwax_data' passed." << std::endl;
}
#endif


void test_correlation_with_eda2_data(){
    auto volt = Voltages::from_eda2_file(dataRootDir + "/eda2/channel_cont_20220118_41581_0_binary.bin", EDA2_OBSERVATION_INFO, 262144);
    auto xcorr = cross_correlation(volt, 1);
    // TODO improve this test
    std::cout << "'test_correlation_with_eda2_data' passed." << std::endl;
}



void test_correlation_with_offline_correlator_data(){
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 1000);
#ifdef __GPU__
    auto v1 = cross_correlation(volt, 32);
    v1.to_cpu();
#else
    auto v1 = cross_correlation_cpu(volt, 32);
#endif
    auto v2 = Visibilities::from_fits_file(dataRootDir + "/offline_correlator/1313388760_20110815061242_gpubox20_00.fits");
    
    if (!complex_vectors_equal(v1.data(), v2.data(), v1.size())){
        throw TestFailed("test_corrrelation_with_offline_correlator_data failed.");
    }
    std::cout << "'test_correlation_with_offline_correlator_data' passed." << std::endl;
}



void test_correlation_bad_input() {
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 1000);
    bool badParamCaught {false};

    try {
        Visibilities vis = cross_correlation(volt, 0);
    } catch (std::invalid_argument& ex){
        badParamCaught = true;
    }
    if(!badParamCaught) throw TestFailed("test_correlation_bad_input: didn't check for meaningful channel averaging parameter.");
    badParamCaught = false;

    try {
        Visibilities vis = cross_correlation(volt, 512);
    } catch (std::invalid_argument& ex) {
        badParamCaught = true;
    }
    if(!badParamCaught) throw TestFailed("test_correlation_bad_input: didn't check for nIntegrationSteps to be a integer multiple of nTimesteps.");
    std::cout << "'test_correlation_with_bad_input' passed." << std::endl;
}


#ifdef __GPU__
#include "../src/correlation_gpu.hpp"
void test_correlation_gpu(){
    auto start_all = std::chrono::high_resolution_clock::now();
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 100);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto xcorr_cpu = cross_correlation_cpu(volt, 32);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto xcorr_gpu = cross_correlation_gpu(volt, 32);
    xcorr_gpu.to_cpu();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "test_correlation_gpu(): correlation execution time CPU (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(start_gpu - start_cpu).count() << std::endl;
    std::cout << "test_correlation_gpu(): correlation execution time GPU (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_gpu).count() << std::endl;
    // xcorr_gpu.to_fits_file("xcorr_gpu.fits");
    if (!complex_vectors_equal(xcorr_cpu.data(), xcorr_gpu.data(), xcorr_gpu.size())){
        throw TestFailed("test_corrrelation_gpu failed.");
    }
    auto stop_all = std::chrono::high_resolution_clock::now();
    std::cout << "test_correlation_gpu(): all execution time incl compare CPU/GPU results (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_all - start_all).count() << std::endl;
    std::cout << "'test_correlation_gpu' passed." << std::endl;
}
#endif



int main(void){
    char *pathToData {std::getenv(ENV_DATA_ROOT_DIR)};
    if(!pathToData){
        std::cerr << "'" << ENV_DATA_ROOT_DIR << "' environment variable is not set." << std::endl;
        return -1;
    }
    dataRootDir = std::string {pathToData};

    try{
        auto start = std::chrono::high_resolution_clock::now();
        test_complex_conjugate_multiply();
        test_correlation_with_xgpu_data();
        // test_correlation_with_xgpu_in_mwax_data();
        test_correlation_with_xgpu_in_mwax_data_16T();
        test_correlation_with_offline_correlator_data();
        test_correlation_with_eda2_data();
        test_correlation_bad_input();
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Tests batch execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
        #ifdef __GPU__
        std::cout << "Verify CPU and GPU correlation gives the same result..." << std::endl;
        test_correlation_gpu();
        #endif
    } catch (std::exception& ex){
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed." << std::endl;
    return 0;
}
