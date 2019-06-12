#include <utils.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_ms_filter/cuda_ms_filter.h>
#include <driver_types.h>

__device__ inline float _get_element(const cudaTextureObject_t img, int row, int col, int width, int height, int chn,
                                     float default_value = -999999.0f) {
    if (row >= 0 && row < height && col >= 0 && col < width) {
        return tex2D<float>(img, col, row + height * chn);
    } else {
        return default_value;
    }
}

template<int blk_w, int blk_h, int ch, int dis_range = 5, int max_iter = 5>
__global__ void _ms_filter(cudaTextureObject_t in_tex,
                           float *output,
                           int width,
                           int height,
                           int pitch,
                           float color_range) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int j = blk_x_idx + threadCol;
    int i = blk_y_idx + threadRow;
    if (j >= width || i >= height)
        return;
    int ic = i;
    int jc = j;
    int ic_old, jc_old;
    float L_old, U_old, V_old;
    float L = tex2D<float>(in_tex, j, i + height * 0);
    float U = tex2D<float>(in_tex, j, i + height * 1);
    float V = tex2D<float>(in_tex, j, i + height * 2);
    L = L * 100 / 255;
    U = U - 128;
    V = V - 128;

    float shift = 5.0f;
    int i2_from = max(0, i - dis_range), i2to = min(height, i + dis_range + 1);
    int j2_from = max(0, j - dis_range), j2to = min(width, j + dis_range + 1);
    __shared__ float neighbor_pixels[ch][
            dis_range + blk_h + dis_range][dis_range + blk_w + dis_range];
#pragma unroll
    for (int r = threadRow - dis_range; r < dis_range + blk_h; r += blk_h) {
        for (int c = threadCol - dis_range; c < blk_w + dis_range; c += blk_w) {
            if (r < dis_range + blk_h && c < dis_range + blk_w) {
                for (int chn = 0; chn < ch; chn++) {
                    neighbor_pixels[chn][c + dis_range][r + dis_range] =
                            _get_element(in_tex, blk_y_idx + r, blk_x_idx + c, width, height, chn);
//                    printf("(%d, %d): %f\n", r, c,
//                           _get_element(in_tex, blk_y_idx + r, blk_x_idx + c, width, height, chn));
                }
            }
        }
    }
    __syncthreads();
    for (int iters = 0; shift > 3 && iters < max_iter; iters++) {
        ic_old = ic;
        jc_old = jc;
        L_old = L;
        U_old = U;
        V_old = V;

        float mi = 0.0f;
        float mj = 0.0f;
        float mL = 0.0f;
        float mU = 0.0f;
        float mV = 0.0f;
        int num = 0.0f;

#pragma unroll 20
        for (int i2 = i2_from; i2 < i2to; i2++) {
            for (int j2 = j2_from; j2 < j2to; j2++) {
                float L2 = neighbor_pixels[0][j2 - blk_x_idx + dis_range][i2 - blk_y_idx + dis_range],
                        U2 = neighbor_pixels[1][j2 - blk_x_idx + dis_range][i2 - blk_y_idx + dis_range],
                        V2 = neighbor_pixels[2][j2 - blk_x_idx + dis_range][i2 - blk_y_idx + dis_range];
//                if (i == 0 && j == 0) {
//                    printf("%f %f %f %d %d %f \n", L2, U2, V2, j2, i2, tex2D<float>(in_tex, 0, 0 + height * 2));
//                }
                L2 = L2 * 100 / 255;
                U2 = U2 - 128;
                V2 = V2 - 128;
                float dL = L2 - L;
                float dU = U2 - U;
                float dV = V2 - V;
                if (dL * dL + dU * dU + dV * dV <= color_range) {
                    mi += i2;
                    mj += j2;
                    mL += L2;
                    mU += U2;
                    mV += V2;
                    num++;
                }
            }
        }
        float num_ = 1.f / num;

        L = mL * num_;
        U = mU * num_;
        V = mV * num_;
        ic = lround(mi * num_ + 0.5);
        jc = lround(mj * num_ + 0.5);
        int di = ic - ic_old;
        int dj = jc - jc_old;
        float dL = L - L_old;
        float dU = U - U_old;
        float dV = V - V_old;

        shift = di * di + dj * dj + dL * dL + dU * dU + dV * dV;
    }

    L = L * 255 / 100;
    U = U + 128;
    V = V + 128;
    output[(i + height * 0) * pitch / sizeof(float) + j] = L;
    output[(i + height * 1) * pitch / sizeof(float) + j] = U;
    output[(i + height * 2) * pitch / sizeof(float) + j] = V;
}

namespace CuMeanShift {
    template<int blk_w, int blk_h, int ch>
    void CudaMsFilter<blk_w, blk_h, ch>::ms_filter_luv(float *input,
                                                       float *output,
                                                       int width,
                                                       int height,
                                                       int pitch,
                                                       int dis_range,
                                                       float color_range,
                                                       int max_iter) {
        dim3 block(blk_w, blk_h);
        dim3 grid(CEIL(width, blk_w), CEIL(height, blk_h));

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
        _ms_filter<blk_w, blk_h, ch, 5, 5> << < grid, block >> >
                                                      (in_tex, output, width, height, pitch, color_range);
        cudaDeviceSynchronize();
        cudaDestroyTextureObject(in_tex);
    }
}