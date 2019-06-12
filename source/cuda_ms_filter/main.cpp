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
 *         third argument: color range
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
    int img_size = img.height() * img.width() * sizeof(float3);

    float *cuda_input, *cuda_output, *output;
    size_t pitch;
    cudaMalloc((void **) &cuda_output, img.width() * img.height() * img.spectrum() * sizeof(float));
    output = (float *) malloc(img.width() * img.height() * img.spectrum() * sizeof(float));
    cudaMallocPitch(
            &cuda_input,
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
    CudaMsFilter<16, 16, 3> uf;
    uf.ms_filter_luv(cuda_input, cuda_output, img.width(), img.height(), pitch, spatial_radius, color_radius, 5, 5);
    cudaMemcpy(output, cuda_output, img.width() * img.height() * img.spectrum() * sizeof(float),
               cudaMemcpyDeviceToHost);
    CImg<float> labels_img(output, img.width(), img.height(), 1, 3);
    CImgDisplay disp(labels_img, "labels", 2);
//    disp.show();
//    while (!disp.is_closed()) {
//        disp.wait();
//    }
//    for (int i = 0; i < min(img.height(), 10); ++i) {
//        for (int j = 0; j < min(img.with(), 10); ++j) {
//            printf("%f %f %f\n", img(j, i, 0), img(j, i, 1), img(j, i, 2));
//        }
//    }
//    if (!outputBin(argv[2], img.size(), output)) {
//        cout << "Failed to output bin file" << endl;
//        return 1;
//    }
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    free(output);
    //cudaMemcpy(labels_h, labels, img.width() * img.height() * sizeof(int), cudaMemcpyDeviceToHost);
    /*
    float3* host_img;
    host_img=(float3*)malloc(img_size);
    for (int i = 0; i < img.height(); ++i) {
        for (int j = 0; j < img.width(); ++j) {
            host_img[i*img.width()+j]=make_float3(img(j,i,0),img(j,i,1),img(j,i,2));
        }
    }
    float3* cuda_img,*cuda_res;
    float3* host_res=(float3*)malloc(img.width()*img.height()* sizeof(float3));
    cudaMalloc((void**)&cuda_img,img_size);
    cudaMalloc((void**)&cuda_res,img_size);
    CudaUnionFind<32, 32, 3, true> uf;
    cudaMemcpy(cuda_img,host_img,img_size,cudaMemcpyHostToDevice);
    uf.union_find(cuda_img, cuda_res,img.width(),img.height(),spatial_radius,color_radius,5,5,host_res);
    printf("hell0 %f %f %f\n", host_res[0].x,host_res[0].y,host_res[0].z);
    printf("hell0 %f %f %f\n", host_res[1].x,host_res[1].y,host_res[1].z);
    for (int i = 0; i < img.height(); ++i) {
        for (int j = 0; j < img.width(); ++j) {
            img(j,i,0)=host_res[i*img.width()+j].x;
            img(j,i,1)=host_res[i*img.width()+j].y;
            img(j,i,2)=host_res[i*img.width()+j].z;
        }
    }
    printf("Res:");
    for (int i = img.height()-1; i > img.height()-10; --i) {
        for (int j = img.width()-1; j > img.width()-10; --j) {
            printf("%f %f %f\n",img(j,i,0),img(j,i,1),img(j,i,2));
        }
    }
    if (!outputBin(argv[2], img.size(), img.data())) {
        cout << "Failed to output bin file" << endl;
        return 1;
    }
    cudaFree(cuda_img);
    cudaFree(cuda_res);*/
    return 0;
}
