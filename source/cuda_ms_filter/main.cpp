#include <CImg.h>
#include <fstream>
#include <iostream>
#include <cuda_ms_filter.h>

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
    if (argc != 5) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 4" << endl;
        return 1;
    }
    CImg<float> img(argv[1]), org_img(img);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    }
    else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }
    long spatial_radius = strtol(argv[3], nullptr, 10);
    float color_radius = strtof(argv[4], nullptr);
    int img_size=img.height()*img.width()*sizeof(float3);
    float3* host_img;
    host_img=(float3*)malloc(img_size);
    for (int i = 0; i < img.height(); ++i) {
        for (int j = 0; j < img.width(); ++j) {
            host_img[i*img.width()+j]=make_float3(img(j,i,0),img(j,i,1),img(j,i,2));
        }
    }
    float3* cuda_img,*cuda_res;
    cudaMalloc((void**)&cuda_img,img_size);
    cudaMalloc((void**)&cuda_res,img_size);
    CudaUnionFind<32, 32, 3, true> uf;
    cudaMemcpy(cuda_img,host_img,img_size,cudaMemcpyHostToDevice);
    uf.union_find(cuda_img, cuda_res,img.width(),img.height(),spatial_radius,color_radius,5,5);
}
