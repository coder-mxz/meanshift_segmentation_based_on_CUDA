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

#define LOOP_TIMES 20

struct Image {
  int width;
  int height;
  int channels;
  int stride;
  float *data;
};

__device__ inline float _get_element(const Image img, int row, int col,
                                     int chn) {
  return img.data[(row + chn * img.height) * img.width + col];
}

template <int channels = 1, int radius = 4, int block_width = 32,
          int block_height = 32>
__global__ void _new_share_flooding(Image img, int *output, float color_range) {
  __shared__ float neighbor_pixels[channels][radius + block_height]
                                  [radius + block_width];
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // load data to share memory
  for (int r = threadRow; r < radius + block_height; r += block_height) {
    for (int c = threadCol; c < radius + block_width; c += block_width) {
      if (r < radius + block_height && c < radius + block_width) {
        for (int chn = 0; chn < channels; chn++) {
          if (blockRow * block_height + r - radius >= 0 &&
              blockRow * block_height + r - radius < img.height &&
              blockCol * block_width + c - radius >= 0 &&
              blockCol * block_width + c - radius < img.width) {
            neighbor_pixels[chn][r][c] =
                _get_element(img, blockRow * block_height + r - radius,
                             blockCol * block_width + c - radius, chn);
          } else {
            neighbor_pixels[chn][r][c] = -999999.0f;
          }
        }
      }
    }
  }
  __syncthreads();
  // process LOOP_TIMES(20) times to flooding image.
  for (int _i = 0; _i < LOOP_TIMES; _i++) {
    if (row < img.height && col < img.width) {
      float min_delta_luv = 999999.0f;
      int min_row = 999999, min_col = 999999;
      for (int r = 0; r < radius; r++) {
        int cur_row = threadRow + r, cur_col = threadCol + r;
        // find up
        if (row - radius + r >= 0) {
          float delta_luv = 0.0f;
          for (int chn = 0; chn < channels; chn++) {
            float delta =
                neighbor_pixels[chn][cur_row][threadCol + radius] -
                neighbor_pixels[chn][threadRow + radius][threadCol + radius];
            delta_luv += delta * delta;
          }
          if (delta_luv < min_delta_luv) {
            min_delta_luv = delta_luv;
            min_row = row - radius + r;
            min_col = col;
          }
        }
        // find left
        if (col - radius + r >= 0) {
          float delta_luv = 0.0f;
          for (int chn = 0; chn < channels; chn++) {
            float delta =
                neighbor_pixels[chn][threadRow + radius][cur_col] -
                neighbor_pixels[chn][threadRow + radius][threadCol + radius];
            delta_luv += delta * delta;
          }
          if (delta_luv < min_delta_luv) {
            min_delta_luv = delta_luv;
            min_row = row;
            min_col = col - radius + r;
          }
        }
      }
      if (min_delta_luv < color_range) {
        output[row * img.width + col] = output[min_row * img.width + min_col];
      }
    }
  }
}

template <int channels = 1, int radius = 4>
__global__ void _new_naive_flooding(Image img, int *output, float color_range) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < img.height) {
    if (col < img.width) {
      output[row * img.width + col] = (row * img.width + col) * 10;
      // printf("(%d, %d): %f\n", row, col, _get_element(img, row, col, 0));
    }
  }
  __syncthreads();
  for (int _i = 0; _i < LOOP_TIMES; _i++) {
    if (row < img.height) {
      if (col < img.width) {
        // process every pixel
        float min_delta_luv = 999999.0f;
        int min_row = 999999, min_col = 999999;
        for (int r = -radius; r < 0; r++) {
          int cur_row = row + r, cur_col = col + r;
          // find up
          if (cur_row >= 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < channels; chn++) {
              float delta = _get_element(img, cur_row, col, chn) -
                            _get_element(img, row, col, chn);
              // img.atXY(col, cur_row, 0, chn) - img.atXY(col, row, 0, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              min_row = cur_row;
              min_col = col;
            }
          }
          // find left
          if (cur_col >= 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < channels; chn++) {
              float delta = _get_element(img, row, cur_col, chn) -
                            _get_element(img, row, col, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              min_row = row;
              min_col = cur_col;
            }
          }
        }
        // set label
        if (min_delta_luv < color_range) {
          output[row * img.width + col] = output[min_row * img.width + min_col];
        }
      }
    }
    __syncthreads();
  }
}

namespace CuMeanShift {
template <int blk_width, int blk_height, int channels, int radius>
void CudaFlooding<blk_width, blk_height, channels, radius>::flooding(
    CImg<float> &img, int *output, float color_range) {
  int *d_output = nullptr;
  for (int x = 0; x < img.width(); x++) {
    for (int y = 0; y < img.height(); y++) {
      output[y * img.width() + x] = (y * img.width() + x) * 10;
    }
  }
  // malloc device labels array
  cudaMalloc((void **)&d_output, sizeof(int) * img.width() * img.height());
  // copy labels to device
  cudaMemcpy(d_output, output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyHostToDevice);
  // malloc device image data array
  float *d_img = nullptr;
  cudaMalloc((void **)&d_img,
             sizeof(float) * img.width() * img.height() * img.spectrum());
  cudaMemcpy(d_img, img.data(),
             sizeof(float) * img.width() * img.height() * img.spectrum(),
             cudaMemcpyHostToDevice);
  Image d_image;
  d_image.channels = channels;
  d_image.stride = d_image.width = img.width();
  d_image.height = img.height();
  d_image.data = d_img;
  // launch kernel
  dim3 block_1(blk_width, blk_height);
  dim3 grid_1(CEIL(img.width(), blk_width), CEIL(img.height(), blk_height));
  _new_share_flooding<channels, radius, blk_width, blk_height>
      <<<grid_1, block_1>>>(d_image, d_output, color_range);
  cudaDeviceSynchronize();
  // copy labels back to host
  cudaMemcpy(output, d_output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyDeviceToHost);
  cudaFree(d_img);
  cudaFree(d_output);
}

// latest naive flooding function, let one thread flood one pixel
template <int channels = 1, int radius = 4>
void _new_cpu_flooding(CImg<float> &img, int *output, float color_range) {
  // initial labels
  for (int row = 0; row < img.height(); row++) {
    for (int col = 0; col < img.width(); col++) {
      output[row * img.width() + col] = (row * img.width() + col) * 10;
    }
  }
  // process LOOP_TIMES times
  for (int _i = 0; _i < LOOP_TIMES; _i++) {
    for (int row = 0; row < img.height(); row++) {
      for (int col = 0; col < img.width(); col++) {
        // process every pixel
        float min_delta_luv = 999999.0f;
        int min_row = 999999, min_col = 999999;
        for (int r = -radius; r < 0; r++) {
          int cur_row = row + r, cur_col = col + r;
          // find up
          if (cur_row >= 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < channels; chn++) {
              float delta =
                  img.atXY(col, cur_row, 0, chn) - img.atXY(col, row, 0, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              min_row = cur_row;
              min_col = col;
            }
          }
          // find left
          if (cur_col >= 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < channels; chn++) {
              float delta =
                  img.atXY(cur_col, row, 0, chn) - img.atXY(col, row, 0, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              min_row = row;
              min_col = cur_col;
            }
          }
        }
        // set label
        if (min_delta_luv < color_range) {
          output[row * img.width() + col] =
              output[min_row * img.width() + min_col];
        }
      }
    }
  }
}

template <int blk_width, int blk_height, int channels, int radius>
void CudaFlooding<blk_width, blk_height, channels, radius>::_test_flooding(
    CImg<float> &img, int *output, float color_range) {
  int *d_output = nullptr;
  // malloc device labels array
  cudaMalloc((void **)&d_output, sizeof(int) * img.width() * img.height());
  // copy labels to device
  cudaMemcpy(d_output, output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyHostToDevice);
  // malloc device image data array
  float *d_img = nullptr;
  cudaMalloc((void **)&d_img,
             sizeof(float) * img.width() * img.height() * img.spectrum());
  cudaMemcpy(d_img, img.data(),
             sizeof(float) * img.width() * img.height() * img.spectrum(),
             cudaMemcpyHostToDevice);
  Image d_image;
  d_image.channels = channels;
  d_image.stride = d_image.width = img.width();
  d_image.height = img.height();
  d_image.data = d_img;

  dim3 block_1(blk_width, blk_height);
  dim3 grid_1(CEIL(img.width(), blk_width), CEIL(img.height(), blk_height));
  _new_share_flooding<channels, radius, blk_width, blk_height>
      <<<grid_1, block_1>>>(d_image, d_output, color_range);
  //_new_naive_flooding<channels, radius>
  //    <<<grid_1, block_1>>>(d_image, d_output, color_range);
  cudaDeviceSynchronize();

  // copy labels back to host
  cudaMemcpy(output, d_output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyDeviceToHost);
  //_new_cpu_flooding<channels, radius>(img, output, color_range);
  cudaFree(d_img);
  cudaFree(d_output);
}
}