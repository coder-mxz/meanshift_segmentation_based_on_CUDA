#include <cstdio>
#include <cstring>
#include <utils.h>
#include <type_traits>
#include <cuda_flooding/cuda_flooding.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#include <driver_types.h>
#endif

__global__ void _initLabels(int *labels, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size)
        return;

    labels[gid] = gid;
}

template <int loop=8>
__global__ void _propagateLabels(int *prop_id, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // flood and propagate labels
    if (gid >= size)
        return;

    for(int i=0; i<loop; i++) {
        prop_id[gid] = prop_id[prop_id[gid]];
    }
}

__global__ void _setLabels(int *labels, int *prop_id, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size)
        return;

    labels[gid] = labels[prop_id[gid]];
}

template<int blk_w = 32, int blk_h = 32,
         int ch = 1, int rad = 4,
         typename std::enable_if<((((rad + blk_h) * blk_w * (ch+1)) +
                                   ((rad + blk_w) * blk_h * (ch+1)) <= 12288)), int>::type = 0>
__global__ void _sharedFlooding(cudaTextureObject_t in_tex,
                                int *labels,
                                int *prop_id,
                                int width,
                                int height,
                                float color_range) {
    __shared__ float left_pixels[ch][blk_h][rad + blk_w];
    __shared__ float up_pixels[ch][rad + blk_h][blk_w];
    int thd_row = threadIdx.y;
    int thd_col = threadIdx.x;
    int blk_row = blockIdx.y * blockDim.y;
    int blk_col = blockIdx.x * blockDim.x;
    int row = blk_row + thd_row;
    int col = blk_col + thd_col;
    int gid = row * width + col;

    if (row >= height || col >= width)
        return;

    // load data to share memory
    for (int r = thd_row; r < rad + blk_h; r += blk_h) {
        for (int chn = 0; chn < ch; chn++) {
            up_pixels[chn][r][thd_col] = tex2D<float>(in_tex, col, (blk_row + r - rad) + chn * height);
        }
    }
    __syncthreads();

    for (int chn = 0; chn < ch; chn++) {
        left_pixels[chn][thd_row][thd_col + rad] = up_pixels[chn][thd_row + rad][thd_col];
    }

    if (thd_col < rad) {
        for (int chn = 0; chn < ch; chn++) {
            left_pixels[chn][thd_row][thd_col] = tex2D<float>(in_tex, col - rad, row + chn * height);
        }
    }
    __syncthreads();


    // find neighbor pixel with the minimum delta LUV
    int offset = 0;
    int new_offset;
    float min_delta_luv = 999999.0f;

    for (int r = 0; r < rad; r++) {
        // find pixel with min(delta LUV) in the left direction
        if (col - rad + r > 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < ch; chn++) {
                float delta = left_pixels[chn][thd_row][thd_col + r] -
                              left_pixels[chn][thd_row][thd_col + rad];
                delta_luv += delta * delta;
            }
            new_offset = row * width + (col - rad + r);;
            if (delta_luv < min_delta_luv) {
                min_delta_luv = delta_luv;
                offset = new_offset;
            }
            else if (delta_luv == min_delta_luv && offset > new_offset) {
                offset = new_offset;
            }
        }

        // find pixel with min(delta LUV) in the up direction
        if (row - rad + r > 0) {
            float delta_luv = 0.0f;
            for (int chn = 0; chn < ch; chn++) {
                float delta = up_pixels[chn][thd_row + r][thd_col] -
                              up_pixels[chn][thd_row + rad][thd_col];
                delta_luv += delta * delta;
            }
            if (delta_luv < min_delta_luv) {
                min_delta_luv = delta_luv;
                offset = (row - rad + r) * width + col;
            }
            else if (delta_luv == min_delta_luv && offset > new_offset) {
                offset = new_offset;
            }
        }

    }

    if (min_delta_luv < color_range) {
        prop_id[gid] = offset;
    }
    else {
        prop_id[gid] = gid;
    }
}

namespace CuMeanShift {
    template<int blk_w, int blk_h, int ch, int rad>
    void CudaFlooding<blk_w, blk_h, ch, rad>::flooding(int *labels,
                                                       float *input,
                                                       int pitch,
                                                       int width,
                                                       int height,
                                                       int loops,
                                                       float color_range) {

        dim3 block(blk_w, blk_h);
        dim3 grid(CEIL(width, blk_w), CEIL(height, blk_h));
        dim3 block2(blk_w * blk_h, 1);
        dim3 grid2(CEIL(width * height, blk_w * blk_h), 1);
        /// create texture object
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = input;
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height * ch;
        res_desc.res.pitch2D.pitchInBytes = pitch;
        res_desc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        res_desc.res.pitch2D.desc.x = 32; // bits per channel

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.filterMode = cudaFilterModePoint;

        cudaTextureObject_t in_tex = 0;
        cudaCreateTextureObject(&in_tex, &res_desc, &tex_desc, NULL);

        int *prop_id;
        int size = width * height;
        cudaMalloc(&prop_id, width * height * sizeof(int));
        _initLabels <<<grid2, block2>>> (labels, size);
        _sharedFlooding<blk_w, blk_h, ch, rad> <<< grid, block >>> (in_tex, labels, prop_id,
                width, height, color_range / loops);
        switch (loops) {
            case 1:
                _propagateLabels<1><<<grid2, block2>>> (prop_id, size);
                break;
            case 2:
                _propagateLabels<2><<<grid2, block2>>> (prop_id, size);
                break;
            case 3:
                _propagateLabels<3><<<grid2, block2>>> (prop_id, size);
                break;
            case 4:
                _propagateLabels<4><<<grid2, block2>>> (prop_id, size);
                break;
            case 5:
                _propagateLabels<5><<<grid2, block2>>> (prop_id, size);
                break;
            case 6:
                _propagateLabels<6><<<grid2, block2>>> (prop_id, size);
                break;
            case 7:
                _propagateLabels<7><<<grid2, block2>>> (prop_id, size);
                break;
            case 8:
                _propagateLabels<8><<<grid2, block2>>> (prop_id, size);
                break;
            case 9:
                _propagateLabels<9><<<grid2, block2>>> (prop_id, size);
                break;
            case 10:
                _propagateLabels<10><<<grid2, block2>>> (prop_id, size);
                break;
            case 11:
                _propagateLabels<11><<<grid2, block2>>> (prop_id, size);
                break;
            case 12:
                _propagateLabels<12><<<grid2, block2>>> (prop_id, size);
                break;
            case 13:
                _propagateLabels<13><<<grid2, block2>>> (prop_id, size);
                break;
            case 14:
                _propagateLabels<14><<<grid2, block2>>> (prop_id, size);
                break;
            case 15:
                _propagateLabels<15><<<grid2, block2>>> (prop_id, size);
                break;
            case 16:
                _propagateLabels<16><<<grid2, block2>>> (prop_id, size);
                break;
            default:
                _propagateLabels<16><<<grid2, block2>>> (prop_id, size);
                break;
        }


        _setLabels<<<grid2, block2>>> (labels, prop_id, size);
        cudaDeviceSynchronize();
        cudaDestroyTextureObject(in_tex);
        cudaFree(prop_id);
    }
}

