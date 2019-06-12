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

__global__ void _helloFromGPU(void) {
  if (threadIdx.x == 5) {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
  }
}
#define BLOCK_SIZE 32
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

__device__ inline void _set_element(Image img, int row, int col, int chn,
                                    float value) {
  if (row >= 0 && row < img.height && col >= 0 && col < img.width) {
    img.data[(row * img.channels + chn) * img.stride + col] = value;
  }
}
template <int block_width = 32, int block_height = 32>
__device__ inline Image _get_sub_image(Image img, int row, int col) {
  Image imgSub;
  imgSub.width = block_width;
  imgSub.height = block_height;
  imgSub.channels = img.channels;
  imgSub.stride = img.stride;
  imgSub.data = &img.data[row * block_height * img.channels * img.stride +
                          col * block_width];
  return imgSub;
}

template <int channels = 1, int radius = 4, int block_width = 32,
          int block_height = 32>
__global__ void _new_flooding(Image img, int *output, float color_range) {
  __shared__ float neighbor_pixels[channels][radius + block_height + radius]
                                  [radius + block_width + radius];
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // thread_id
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // local thread_id
  // int ltid = tid % block_width;
  // process sub image
  for (int ir = 0; ir < CEIL(img.height, block_height); ir++) {
    for (int ic = 0; ic < CEIL(img.width, block_width); ic++) {
      // load data to share memory
      for (int r = threadRow; r < radius + block_height + radius;
           r += block_height) {
        for (int c = threadCol; c < radius + block_width + radius;
             c += block_width) {
          if (r < radius + block_height + radius &&
              c < radius + block_width + radius) {
            for (int chn = 0; chn < channels; chn++) {
              neighbor_pixels[chn][r][c] =
                  _get_element(img, ir * block_height + r - radius,
                               ic * block_width + c - radius, chn);
              //printf("(%d, %d): %f\n", r, c,
              //       _get_element(img, ir * block_height + r - radius,
              //                    ic * block_width + c - radius, chn));
            }
          }
        }
      }
      // set labels
      output[(ir * block_height + threadRow) * img.width + ic * block_width +
             threadCol] = (ir * block_height + threadRow) * img.width +
                          ic * block_width + threadCol;
      __syncthreads();
      // process LOOP_TIMES(20) times to flooding image.
      for (int i = 0; i < LOOP_TIMES; i++) {
        int r = threadRow + radius, c = threadCol + radius;
        float min_delta = 999999.0f;
        int minr = 999999, minc = 999999;
        // compare current pixel to left 4 and up 4 pixel, make current
        // pixel's label equal to min deltaLUV pixel
        for (int j = 0; j < radius; j++) {
          float delta_luv = 0.0f;
          for (int chn = 0; chn < channels; chn++) {
            float delta =
                neighbor_pixels[chn][threadRow + j][threadCol] -
                neighbor_pixels[chn][threadRow + radius][threadCol + radius];
            delta_luv += delta * delta;
          }
          if (min_delta > delta_luv) {
            min_delta = delta_luv;
            minr = threadRow + j;
            minc = threadCol;
          }
          delta_luv = 0.0f;
          for (int chn = 0; chn < channels; chn++) {
            float delta =
                neighbor_pixels[chn][threadRow][threadCol + j] -
                neighbor_pixels[chn][threadRow + radius][threadCol + radius];
            delta_luv += delta * delta;
          }
          if (min_delta > delta_luv) {
            min_delta = delta_luv;
            minr = threadRow;
            minc = threadCol + j;
          }
        }
        if (min_delta < color_range) {
          output[(ir * block_height + threadRow) * img.width +
                 ic * block_width + threadCol] =
              output[(ir * block_height + minr) * img.width + ic * block_width +
                     minc];
        }
      }
      __syncthreads();
    }
  }
}

template <int channels = 1, int radius = 4>
__global__ void _naive_flooding(Image img, int *output, float color_range) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float min_delta = 999999.0f;
  int minr = row;
  int minc = col;
  // printf("(%d, %d)\n", row, col);
  __syncthreads();
  if (row < img.height && col < img.width) {
    output[row * img.width + col] = row * img.width + col;
    // printf("(%d, %d): %f\n", row, col, _get_element(img, row, col, 0));
  }

  __syncthreads();
  if (row < img.height && col < img.width) {
    for (int _i = 0; _i < LOOP_TIMES; _i++) {
      for (int r = -radius; r < 0; r++) {
        float delta_luv = 0.0f;
        for (int chn = 0; chn < channels; chn++) {
          float delta = _get_element(img, row, col, chn) -
                        _get_element(img, row + r, col, chn);
          delta_luv += delta * delta;
        }
        if (delta_luv < min_delta) {
          min_delta = delta_luv;
          minr = row + r;
          minc = col;
        }
        delta_luv = 0.0f;
        for (int chn = 0; chn < channels; chn++) {
          float delta = _get_element(img, row, col, chn) -
                        _get_element(img, row, col + r, chn);
          delta_luv += delta * delta;
        }
        if (delta_luv < min_delta) {
          min_delta = delta_luv;
          minr = row;
          minc = col + r;
        }
      }
      if (min_delta < color_range && minr >= 0 && minr < img.height &&
          minc >= 0 && minc < img.width) {
        output[row * img.width + col] = output[minr * img.width + minc];
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
      //printf("(%d, %d): %f\n", row, col, _get_element(img, row, col, 0));
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

template <int channels = 1, int radius = 4>
__global__ void _flooding(cudaTextureObject_t in_tex, int *output, int width,
                          int height, float color_range) {
  //__shared__ extern float neighbor_pixels[]; //存储邻域像素值
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  output[x * width + y] = x * width + y;
  for (int i = 0; i < channels; i++) {
    if (x >= 0 && x < height && y >= 0 && y < width) {
      float debug_value = tex2D<float>(in_tex, y, x + i * height);
      // neighbor_pixels[x * width + y + i] = debug_value;
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
            // float delta = neighbor_pixels[x * width + y + k] -
            //              neighbor_pixels[xx * width + yy + k];
            // delta_luv += delta * delta;
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

template <int channels = 1, int radius = 4>
void _cpu_flooding(CImg<float> &img, int *output, float color_range) {
  for (int row = 0; row < img.height(); row++) {
    for (int col = 0; col < img.width(); col++) {
      output[row * img.width() + col] = row * img.width() + col;
    }
  }
  for (int _i = 0; _i < LOOP_TIMES; _i++) {
    for (int row = 0; row < img.height(); row++) {
      for (int col = 0; col < img.width(); col++) {
        float min_delta_luv = 999999.0f;
        int minr = row, minc = col;
        for (int r = -radius; r < 0; r++) {
          int rr = row + r, cc = col + r;
          float delta_luv = 0.0f;
          if (rr >= 0 && rr < img.height()) {
            for (int chn = 0; chn < channels; chn++) {
              float delta =
                  img.atXY(col, row, 0, chn) - img.atXY(col, rr, 0, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              minr = rr;
              minc = col;
            }
          }
          delta_luv = 0.0f;
          if (cc >= 0 && cc < img.height()) {
            for (int chn = 0; chn < channels; chn++) {
              float delta =
                  img.atXY(col, row, 0, chn) - img.atXY(cc, row, 0, chn);
              delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
              min_delta_luv = delta_luv;
              minr = row;
              minc = cc;
            }
          }
        }
        if (min_delta_luv < color_range) {
          output[row * img.width() + col] = output[minr * img.width() + minc];
        }
      }
    }
  }
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
  cudaMemset(d_output, 0, sizeof(int) * img.width() * img.height());

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
  //_new_flooding<channels, radius, blk_width, blk_height>
  //    <<<grid_1, block_1>>>(d_image, d_output, color_range);
  //_helloFromGPU<<<1, 10>>>();
  _new_naive_flooding<channels, radius>
      <<<grid_1, block_1>>>(d_image, d_output, color_range);
  cudaDeviceSynchronize();

  // copy labels back to host
  cudaMemcpy(output, d_output, sizeof(int) * img.width() * img.height(),
             cudaMemcpyDeviceToHost);
  //_new_cpu_flooding<channels, radius>(img, output, color_range);
  cudaFree(d_img);
  cudaFree(d_output);
}
}