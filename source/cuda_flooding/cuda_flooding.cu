#include <cuda_flooding/cuda_flooding.h>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

//#define IDX(x, y, s) ((y) * (s) + (x))
__shared__ float *neighbor_pixels; //存储邻域像素值

template <int radius>
__global__ void flooding(int *output, int channels, int pitch, int width,
                         int height, int row_offset, int col_offset,
                         float color_range) {
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
            delta_luv += delta;
          }
          if (delta_luv < color_range) {
            output[xx * width + yy] = x * width + y;
          }
        }
      }
    }
  }
}