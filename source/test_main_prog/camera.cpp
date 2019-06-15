//
// Created by iffi on 19-6-8.
//
#define cimg_use_opencv
#include <CImg.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <cuda_ms_filter/cuda_ms_filter.h>
#include <cuda_union_find/cuda_union_find.h>
#include "visualize.h"

using namespace std;
using namespace cimg_library;
using namespace CuMeanShift;
using namespace spdlog;

int main(int argc, char **argv) {
    CImg<float> img, img_ms, img_res;
    CImg<int> img_labels;
    CImgDisplay display, display2, display3, display4;
    CudaMsFilter<32, 32, 3> ms;
    CudaUnionFind<32, 32, 3, true> uf;
    CudaColorLabels<32, 32> cl;

    auto logger = stdout_color_st("logger");

    /// read parameters
    long spatial_radius = strtol(argv[1], nullptr, 10);
    float color_radius = strtof(argv[2], nullptr);
    float color_radius_uf = strtof(argv[3], nullptr);

    /// initialize camera
    img.load_camera(0, 640, 480, 0, false);
    if ( img.is_empty() ) {
        logger->error("Could not capture from camera 0");
        return 1;
    }
    else
        logger->info("Camera 0 initialized");

    /// initialize memory buffers
    float *image_dev_1, *image_dev_2, *image_dev_3;
    int *labels_dev;
    size_t pitch;
    float *image_host_ms = new float[640 * 480 * 3];
    float *image_host_res = new float[640 * 480 * 3];
    int *labels_host = new int[640 * 480];
    int label_count;
    cudaMallocPitch(&image_dev_1, &pitch, 640 * sizeof(float), 480 * 3);
    cudaMallocPitch(&image_dev_2, &pitch, 640 * sizeof(float), 480 * 3);
    cudaMallocPitch(&image_dev_3, &pitch, 640 * sizeof(float), 480 * 3);
    cudaMalloc(&labels_dev, 640 * 480 * sizeof(int));

    /// begin capture and calculation
    display.show();
    display2.show();

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    while(true) {
        // Load image
        img.load_camera(0, 640, 480, 0, false);

        display.display(img).set_title("captured");
        if (!img_ms.is_empty())
            display2.display(img_ms).set_title("filtered");
        if (!img_res.is_empty())
            display3.display(img_res).set_title("result");
        if (!img_labels.is_empty())
            display4.display(img_labels).set_title("labels").set_normalization(1);
        if ( img.is_empty() || display.is_keyQ() || display.is_keyESC() || display.is_closed())
            break;
        cudaEventRecord(begin);

        cudaMemcpy2D(image_dev_1, pitch, img.data(), 640 * sizeof(float), 640 * sizeof(float), 480 * 3,
                cudaMemcpyHostToDevice);
        ms.ms_filter_luv(image_dev_1, image_dev_2, 640, 480, pitch, spatial_radius, color_radius);
        uf.union_find(nullptr, image_dev_2, labels_dev, &label_count, pitch, 640, 480, color_radius_uf);
        cl.color_labels(labels_dev, image_dev_3, label_count, pitch, 640, 480);
        cudaMemcpy2D(image_host_ms, 640 * sizeof(float), image_dev_2, pitch, 640 * sizeof(float), 480 * 3,
                     cudaMemcpyDeviceToHost);
        cudaMemcpy2D(image_host_res, 640 * sizeof(float), image_dev_3, pitch, 640 * sizeof(float), 480 * 3,
                     cudaMemcpyDeviceToHost);
        cudaMemcpy(labels_host, labels_dev, 640 * 480 * sizeof(int), cudaMemcpyDeviceToHost);
        img_ms.assign(image_host_ms, 640, 480, 1, 3, true);
        img_res.assign(image_host_res, 640, 480, 1, 3, true);
        img_labels.assign(labels_host, 640, 480, 1, 1, true);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float time;
        cudaEventElapsedTime(&time, begin, end);
        logger->info("Time: {:.2f} ms | Regions: {:d}", time, label_count);
    }

    // Release camera
    img.load_camera(0, 640, 480, 0, true);
    logger->info("Camera released");

    cudaFree(image_dev_1);
    cudaFree(image_dev_2);
    cudaFree(image_dev_3);

    delete [] image_host_ms;
    delete [] image_host_res;
    delete [] labels_host;
    logger->info("Resource freed");
}
