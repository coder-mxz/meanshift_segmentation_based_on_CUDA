#include <cuda_union_find.h>

texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入

template <int blk_w, int blk_h, int ch>
__global__ void _local_union_find(int *labels, int pitch, int width, int height, float range) {
    __shared__ int blk_labels[blk_w * blk_h];
    __shared__ float blk_pixels[blk_w * blk_h * ch];

    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int blk_size = blk_w * blk_h;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int tid = blk_y_idx * blk_w + blk_x_idx;
    int gid = col * width + col;
    int p_ele = pitch / sizeof(float);

    if (row >= height || col >= width)
        return;

    /// read labels
    blk_labels[tid] = labels[gid];

    /// read pixels
    for(int i=0; i<ch; i++)
        blk_pixels[tid + i * blk_size] = tex2D(in_tex, col, row);

    __syncthreads();

    /// row scan
    float diff, diff_sum = 0;

    if (blk_x_idx != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - 1 + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            blk_labels[tid] = blk_labels[tid - 1];
    }
    __syncthreads();

    diff_sum = 0;
    /// column scan
    if (blk_y_idx != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - blk_w + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            blk_labels[tid] = blk_labels[tid - blk_w];
    }
    __syncthreads();

    /// row-column unification

}

namespace CuMeanShift {
    template <int blk_w, int blk_h, int ch>
    void CudaUnionFind<blk_w, blk_h, ch>::union_find(int *labels,
                                                     float *input,
                                                     int *new_labels,
                                                     int *label_counter,
                                                     int pitch,
                                                     int width,
                                                     int height) {

    }
}