#include <cuda_flooding/cuda_flooding.h>
#include <utils.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

// texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入
//#define IDX(x, y, s) ((y) * (s) + (x))

template <int channels = 1, int radius = 4>
__global__ void flooding(cudaTextureObject_t in_tex, int *output, int width,
                         int height, float color_range) {
  __shared__ float *neighbor_pixels; //存储邻域像素值
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  output[x * width + y] = x * width + y;
  for (int i = 0; i < channels; i++) {
    neighbor_pixels[x * width + y + i] = tex2D(in_tex, x, y + i);
  }
  __syncthreads();
  for (int _count = 0; _count < 20; _count++) {
    for (int i = -radius; i <= radius; i++) {
      for (int j = -radius; j <= radius; j++) {
        int xx = x + i, yy = y + j;
        if (xx >= 0 && xx < height && yy >= 0 && yy < width) {
          float delta_luv = 0.0f;
          for (int k = 0; k < channels; k++) {
            float delta = neighbor_pixels[x * width + y + k] -
                          neighbor_pixels[xx * width + yy + k];
            delta_luv += delta * delta;
          }
          if (delta_luv < color_range) {
            output[xx * width + yy] = x * width + y;
          }
        }
      }
    }
  }
}

namespace CuMeanShift {
template <int blk_width, int blk_height, int channels, int radius>
void CudaFlooding<blk_width, blk_height, channels, radius>::flooding(
    cudaTextureObject_t in_tex, int *output, int width, int height,
    float color_range) {
  dim3 block_1(blk_width, blk_height);
  dim3 grid_1(CEIL(width, blk_width), CEIL(height, blk_height));
  flooding<channels, radius>
      <<<grid_1, block_1, sizeof(float) * width * height * channels>>>(
          in_tex, output, width, height);
  cudaDeviceSynchronize();
}
}