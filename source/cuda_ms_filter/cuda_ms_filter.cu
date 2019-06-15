#include <utils.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_ms_filter/cuda_ms_filter.h>
#include <driver_types.h>

template<int blk_w, int blk_h, int dis_range = 5, int max_iter = 5,
        typename std::enable_if<(((blk_w + 2 * dis_range) * (blk_h + 2 * dis_range) * 3 <= 12288)), int>::type = 0>
__global__ void _msFilterLUV(cudaTextureObject_t in_tex,
                           float *output,
                           int width,
                           int height,
                           int pitch,
                           float color_range) {
    __shared__ float pixels[3][(blk_w + 2 * dis_range) * (blk_h + 2 * dis_range)];

    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;

    if (row >= height || col >= width)
        return;

    /// read pixel values into the shared memory
    int start_col = blk_x_idx - dis_range;
    int start_row = blk_y_idx - dis_range;

    const int patch_width = blk_w + 2 * dis_range;
    const int patch_height = blk_h + 2 * dis_range;
    const int patch_cols = (patch_width + (blk_w - 1)) / blk_w;
    const int patch_rows = (patch_height + (blk_h - 1)) / blk_h;

    const int height2 = height * 2;
#pragma unroll
    for(int i=0; i<patch_rows; i++) {
#pragma unroll
        for(int j=0; j<patch_cols; j++) {
            const int p_col = j * blk_w + threadIdx.x;
            const int p_row = i * blk_h + threadIdx.y;
            const int read_col = start_col + p_col;
            const int read_row = start_row + p_row;
            if (p_col < patch_width && p_row < patch_height) {
                const int p_offset = p_row * patch_width + p_col;
                pixels[0][p_offset] = tex2D<float>(in_tex, read_col, read_row) * 100 / 255;
                pixels[1][p_offset] = tex2D<float>(in_tex, read_col, read_row + height) - 128;
                pixels[2][p_offset] = tex2D<float>(in_tex, read_col, read_row + height2) - 128;
            }
        }
    }

    __syncthreads();

    int cur_row = threadIdx.y + dis_range;
    int cur_col = threadIdx.x + dis_range;
    int old_row, old_col;
    float old_L, old_U, old_V;

    float L = pixels[0][cur_row * patch_width + cur_col];
    float U = pixels[1][cur_row * patch_width + cur_col];
    float V = pixels[2][cur_row * patch_width + cur_col];

    float shift = 5.0f;
    int col_from = threadIdx.x, col_to = col_from + 2 * dis_range + 1;
    int row_from = threadIdx.y, row_to = row_from + 2 * dis_range + 1;

    for (int iters = 0; shift > 3 && iters < max_iter; iters++) {

        old_row = cur_row;
        old_col = cur_col;
        old_L = L;
        old_U = U;
        old_V = V;

        float avg_row = 0.0f;
        float avg_col = 0.0f;
        float avg_L = 0.0f;
        float avg_U = 0.0f;
        float avg_V = 0.0f;
        int num = 0;

#pragma unroll
        for (int r = row_from; r < row_to; r++) {
//#pragma unroll
            for (int c = col_from; c < col_to; c++) {
                float L2 = pixels[0][r * patch_width + c],
                        U2 = pixels[1][r * patch_width + c],
                        V2 = pixels[2][r * patch_width + c];

                float d_L = L2 - L;
                float d_U = U2 - U;
                float d_V = V2 - V;

                if (d_L * d_L + d_U * d_U + d_V * d_V <= color_range) {
                    avg_row += r;
                    avg_col += c;
                    avg_L += L2;
                    avg_U += U2;
                    avg_V += V2;
                    num++;
                }

            }
        }


        float num_ = 1.f / num;

        L = avg_L * num_;
        U = avg_U * num_;
        V = avg_V * num_;
        cur_row = lround(avg_row * num_ + 0.5);
        cur_col = lround(avg_col * num_ + 0.5);
        int d_row = cur_row - old_row;
        int d_col = cur_col - old_col;
        float d_L = L - old_L;
        float d_U = U - old_U;
        float d_V = V - old_V;

        shift = d_row * d_row + d_col * d_col + \
                d_L * d_L + d_U * d_U + d_V * d_V;

    }

    L = L * 255 / 100;
    U = U + 128;
    V = V + 128;

    const int pitch_ele = pitch / sizeof(float);

    output[row * pitch_ele + col] = L;
    output[(row + height * 1) * pitch_ele + col] = U;
    output[(row + height * 2) * pitch_ele + col] = V;
}

namespace CuMeanShift {
    template<int blk_w, int blk_h, int ch>
    void CudaMsFilter<blk_w, blk_h, ch>::msFilterLUV(float *input,
                                                       float *output,
                                                       int width,
                                                       int height,
                                                       int pitch,
                                                       int dis_range,
                                                       float color_range) {
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
        switch (dis_range) {
            case 1:
                _msFilterLUV<blk_w, blk_h, 1, 5> << < grid, block >> >
                                                               (in_tex, output, width, height, pitch, color_range);
                break;
            case 2:
                _msFilterLUV<blk_w, blk_h, 2, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 3:
                _msFilterLUV<blk_w, blk_h, 3, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 4:
                _msFilterLUV<blk_w, blk_h, 4, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 5:
                _msFilterLUV<blk_w, blk_h, 5, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 6:
                _msFilterLUV<blk_w, blk_h, 6, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 7:
                _msFilterLUV<blk_w, blk_h, 7, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 8:
                _msFilterLUV<blk_w, blk_h, 8, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 9:
                _msFilterLUV<blk_w, blk_h, 9, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 10:
                _msFilterLUV<blk_w, blk_h, 10, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 11:
                _msFilterLUV<blk_w, blk_h, 11, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 12:
                _msFilterLUV<blk_w, blk_h, 12, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 13:
                _msFilterLUV<blk_w, blk_h, 13, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 14:
                _msFilterLUV<blk_w, blk_h, 14, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 15:
                _msFilterLUV<blk_w, blk_h, 15, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            case 16:
                _msFilterLUV<blk_w, blk_h, 16, 5> << < grid, block >> >
                                                              (in_tex, output, width, height, pitch, color_range);
                break;
            default:
                break;
        }

        cudaDeviceSynchronize();
        cudaDestroyTextureObject(in_tex);
    }
}