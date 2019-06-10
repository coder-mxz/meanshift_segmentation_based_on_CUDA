#include <CImg.h>
#include <cstdio>
#include <cstring>
#include <cuda_flooding/cuda_flooding.h>
#include <utils.h>
using namespace cimg_library;

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#include <driver_types.h>
#endif

// texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入
//#define IDX(x, y, s) ((y) * (s) + (x))

template <int channels = 1, int radius = 4>
__global__ void _flooding(cudaTextureObject_t in_tex, int *output, int width,
                          int height, float color_range) {
  __shared__ extern float neighbor_pixels[]; //存储邻域像素值
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  output[x * width + y] = x * width + y;
  for (int i = 0; i < channels; i++) {
    if (x >= 0 && x < height && y >= 0 && y < width) {
      float debug_value = tex2D<float>(in_tex, y, x + i * height);
      neighbor_pixels[x * width + y + i] = debug_value;
      if (debug_value != 0.0f) {
        printf("(%d, %d): %f\n", x, y, debug_value);
      }
    }
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
            printf("%f\n", delta_luv);
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
  _flooding<channels, radius>
      <<<grid_1, block_1, sizeof(float) * width * height * channels>>>(
          in_tex, output, width, height, color_range);
  cudaDeviceSynchronize();
}

template <int blk_width, int blk_height, int channels, int radius>
void CudaFlooding<blk_width, blk_height, channels, radius>::_test_flooding(
    CImg<float> &img, int *output, float color_range) {
  int *d_output = nullptr;
  cudaMalloc((void **)&d_output, sizeof(int) * img.width() * img.height());
  cudaMemset(d_output, 0, sizeof(int) * img.width() * img.height());

  /*for (int x = 0; x < img.width(); x++) {
    for (int y = 0; y < img.height(); y++) {
      float debug_value = img.atXY(x, y, 0, 0);
      if (debug_value != 0.0f) {
        printf("(%d, %d): %f\n", x, y, debug_value);
      }
    }
  }*/

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray *arr;
  cudaMallocArray(&arr, &desc, img.width(), img.height());
  cudaMemcpyToArray(arr, 0, 0, img.data(),
                    sizeof(float) * img.width() * img.height(),
                    cudaMemcpyHostToDevice);

  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = arr;

  cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.addressMode[0] = cudaAddressModeBorder;
  tex_desc.addressMode[1] = cudaAddressModeBorder;
  tex_desc.filterMode = cudaFilterModePoint;

  cudaTextureObject_t in_tex = 0;
  cudaCreateTextureObject(&in_tex, &res_desc, &tex_desc, NULL);

  flooding(in_tex, d_output, img.width(), img.height(), color_range);
  cudaDeviceSynchronize();
  cudaDestroyTextureObject(in_tex);
  cudaFreeArray(arr);
  cudaMemcpy(output, d_output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyDeviceToHost);
  cudaFree(d_output);
}
}