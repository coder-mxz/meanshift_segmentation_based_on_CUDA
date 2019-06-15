//
// Created by iffi on 19-6-8.
//

#include <CImg.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <cuda_ms_filter/cuda_ms_filter.h>
#include <cuda_flooding/cuda_flooding.h>
#include <cuda_union_find/cuda_union_find.h>
#include "visualize.h"

using namespace std;
using namespace cimg_library;
using namespace CuMeanShift;
using namespace spdlog;

int main(int argc, char **argv) {
    CImg<float> img(argv[1]), img_ms, img_res;
    CImg<int> img_labels;
    CImgDisplay display, display2, display3, display4;
    CudaMsFilter<32, 32, 3> ms;
    CudaFlooding<32, 32, 3, 4> f;
    CudaUnionFind<32, 32, 3, true> uf;
    CudaColorLabels<32, 32> cl;

    auto logger = stdout_color_st("logger");

    /// read parameters
    long spatial_radius = strtol(argv[2], nullptr, 10);
    float color_radius = strtof(argv[3], nullptr);
    float color_radius_uf = strtof(argv[4], nullptr);

    logger->info("Initialized");

    /// initialize memory buffers
    float *image_dev_1, *image_dev_2, *image_dev_3;
    int *labels_dev, *labels_dev2;
    size_t pitch;
    float *image_host_ms = new float[img.width() * img.height() * 3];
    float *image_host_res = new float[img.width() * img.height() * 3];
    int *labels_host = new int[img.width() * img.height()];
    int label_count;
    cudaMallocPitch(&image_dev_1, &pitch, img.width() * sizeof(float), img.height() * 3);
    cudaMallocPitch(&image_dev_2, &pitch, img.width() * sizeof(float), img.height() * 3);
    cudaMallocPitch(&image_dev_3, &pitch, img.width() * sizeof(float), img.height() * 3);
    cudaMalloc(&labels_dev, img.width() * img.height() * sizeof(int));
    cudaMalloc(&labels_dev2, img.width() * img.height() * sizeof(int));
    /// begin capture and calculation
    display.show();

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);

    cudaMemcpy2D(image_dev_1, pitch, img.data(), img.width() * sizeof(float), img.width() * sizeof(float), img.height() * 3,
                 cudaMemcpyHostToDevice);
    ms.msFilterLUV(image_dev_1, image_dev_2, img.width(), img.height(), pitch, spatial_radius, color_radius);
    f.flooding(labels_dev2, image_dev_2, pitch, img.width(), img.height(), 6, color_radius_uf);
    uf.unionFind(labels_dev2, image_dev_2, labels_dev, &label_count, pitch, img.width(), img.height(), color_radius_uf);
    cl.colorLabels(labels_dev, image_dev_3, label_count, pitch, img.width(), img.height());
    cudaMemcpy2D(image_host_ms, img.width() * sizeof(float), image_dev_2, pitch, img.width() * sizeof(float), img.height() * 3,
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(image_host_res, img.width() * sizeof(float), image_dev_3, pitch, img.width() * sizeof(float), img.height() * 3,
                 cudaMemcpyDeviceToHost);
    cudaMemcpy(labels_host, labels_dev, img.width() * img.height() * sizeof(int), cudaMemcpyDeviceToHost);
    img_ms.assign(image_host_ms, img.width(), img.height(), 1, 3, true);
    img_res.assign(image_host_res, img.width(), img.height(), 1, 3, true);
    img_labels.assign(labels_host, img.width(), img.height(), 1, 1, true);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    logger->info("Time: {:.2f} ms | Regions: {:d}", time, label_count);

    while(true) {
        display.display(img).set_title("captured");
        display2.display(img_ms).set_title("filtered");
        display3.display(img_res).set_title("result");
        display4.display(img_labels).set_title("labels").set_normalization(1);
        if (img.is_empty() || display.is_keyQ() || display.is_keyESC() || display.is_closed())
            break;
    }


    cudaFree(image_dev_1);
    cudaFree(image_dev_2);
    cudaFree(image_dev_3);

    cudaFree(labels_dev);
    cudaFree(labels_dev2);

    delete [] image_host_ms;
    delete [] image_host_res;
    delete [] labels_host;
    logger->info("Resource freed");
}
