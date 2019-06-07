#include <CImg.h>
#include <fstream>
#include <iostream>
#include <cuda_union_find.h>

using namespace std;
using namespace CuMeanShift;
using namespace cimg_library;

bool outputBin(const char *path, size_t count, int *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 *         third argument: color range
 */
int main(int argc, char *argv[])
{
    if (argc != 4)
        return 1;
    CudaUnionFind<32, 32, 3, true> uf;
    /// read test image
    CImg<float> img(argv[1]);

    if (img.spectrum() != 3)
        return 1;

    float *img_dev_data;
    float range = strtof(argv[3], nullptr);

    int *labels, *labels_h;
    int label_count;
    size_t pitch;
    labels_h = new int[img.width() * img.height()];
    cudaMalloc(&labels, img.width() * img.height() * sizeof(int));
    cudaMallocPitch(
            &img_dev_data,
            &pitch,
            img.width() * sizeof(float),
            img.height() * img.spectrum());
    cudaMemcpy2D(
            img_dev_data,
            pitch,
            img.data(),
            img.width() * sizeof(float),
            img.width() * sizeof(float),
            img.height() * img.spectrum(),
            cudaMemcpyHostToDevice);
    uf.union_find(nullptr, img_dev_data, labels, &label_count, pitch, img.width(), img.height(), range);
    cudaMemcpy(labels_h, labels, img.width() * img.height() * sizeof(int), cudaMemcpyDeviceToHost);

    CImg<int> labels_img(labels_h, img.width(), img.height());
    CImgDisplay disp(labels_img, "filtered", 0);
    disp.show();
    while (!disp.is_closed()) {
        disp.wait();
    }

    if (!outputBin(argv[2], img.width() * img.height(), labels_h)) {
        cout << "Failed to output bin file" << endl;
    }

    delete [] labels_h;
    cudaFree(labels);
    cudaFree(img_dev_data);
}
