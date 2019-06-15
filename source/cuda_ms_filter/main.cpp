#include <CImg.h>
#include <fstream>
#include <iostream>
#include <cuda_ms_filter.h>

typedef cudaTextureObject_t i;
using namespace std;
using namespace CuMeanShift;
using namespace cimg_library;

bool outputBin(const char *path, size_t count, int *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char *) &count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 *         third argument: spatial radius
 *         fourth argument: color radius
 */
bool outputBin(const char *path, size_t count, float *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char *) &count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(float));
    file.close();
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 4" << endl;
        return 1;
    }
    CImg<float> img(argv[1]);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    } else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }
    long spatial_radius = strtol(argv[3], nullptr, 10);
    float color_radius = strtof(argv[4], nullptr);

    float *cuda_input, *cuda_output, *output;
    size_t pitch;
    output = (float *) malloc(img.width() * img.height() * img.spectrum() * sizeof(float));
    cudaMallocPitch(
            &cuda_input,
            &pitch,
            img.width() * sizeof(float),
            img.height() * img.spectrum());
    cudaMallocPitch(
            &cuda_output,
            &pitch,
            img.width() * sizeof(float),
            img.height() * img.spectrum());
    cudaMemcpy2D(
            cuda_input,
            pitch,
            img.data(),
            img.width() * sizeof(float),
            img.width() * sizeof(float),
            img.height() * img.spectrum(),
            cudaMemcpyHostToDevice);
    CudaMsFilter<32, 32, 3> ms;
    ms.ms_filter_luv(cuda_input, cuda_output, img.width(), img.height(), pitch, spatial_radius, color_radius);
    cudaMemcpy2D(output,
            img.width() * sizeof(float),
            cuda_output,
            pitch,
            img.width() * sizeof(float),
            img.height() * img.spectrum(),
            cudaMemcpyDeviceToHost);

    CImg<float> filtered_img(output, img.width(), img.height(), 1, 3);
    CImgDisplay disp(filtered_img, "filtered", 2);
    disp.show();
    while (!disp.is_closed()) {
        disp.wait();
    }

    if (!outputBin(argv[2], filtered_img.size(), output)) {
        cout << "Failed to output bin file" << endl;
        return 1;
    }
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    free(output);

    return 0;
}
