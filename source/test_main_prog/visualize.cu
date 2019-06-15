#include "visualize.h"
#include <utils.h>
__global__ void _colorLabels(int *labels,
                              float *image,
                              int label_count,
                              int pitch,
                              int width,
                              int height) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int pitch_ele = pitch / sizeof(float);
    int idx = row * pitch_ele + col;
    int offset = width * height;

    int label = labels[row * width + col];

    unsigned int tmp = label * 0x01234567;

    image[idx] = (tmp & 0xFF0000) >> 4;
    image[idx + offset] = (tmp & 0xFF00) >> 2;
    image[idx + offset * 2] = tmp & 0xFF;
}

namespace CuMeanShift {
    template <int blk_w, int blk_h>
    void CudaColorLabels<blk_w, blk_h>::colorLabels(int *labels,
                          float *image,
                          int label_count,
                          int pitch,
                          int width,
                          int height) {
        dim3 block_1(blk_w, blk_h);
        dim3 grid_1(CEIL(width, blk_w), CEIL(height, blk_h));
        _colorLabels<<<grid_1, block_1>>>(labels, image, label_count, pitch, width, height);
        cudaDeviceSynchronize();
    }
}

template class CuMeanShift::CudaColorLabels<16, 16>;
template class CuMeanShift::CudaColorLabels<32, 32>;
