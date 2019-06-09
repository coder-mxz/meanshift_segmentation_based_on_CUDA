#include <utils.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_ms_filter/cuda_ms_filter.h>
#include <driver_types.h>
#define DEBUG
__device__ void _union_find(int *labels, int idx_1, int idx_2) {
    while (1) {
        idx_1 = labels[idx_1];
        idx_2 = labels[idx_2];
        if (idx_1 < idx_2)
            atomicMin(&labels[idx_2], idx_1);
        else if (idx_1 > idx_2)
            atomicMin(&labels[idx_1], idx_2);
        else
            break;
    }
}

template<int blk_w, int blk_h, int ch, bool ignore_labels = false>
__global__ void _local_union_find(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    __shared__ int blk_labels[blk_w * blk_h];
    __shared__ float blk_pixels[blk_w * blk_h * ch];

    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int blk_size = blk_w * blk_h;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int tid = blk_y_idx * blk_w + blk_x_idx;
    int gid = col * width + col;

    if (row >= height || col >= width)
        return;

    /// read labels
    if (ignore_labels)
        blk_labels[tid] = gid;
    else
        blk_labels[tid] = labels[gid];

    /// read pixels
    for (int i = 0; i < ch; i++)
        blk_pixels[tid + i * blk_size] = tex2D<float>(in_tex, col, row * i);

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
    int tmp_label = tid;
    while (tmp_label != blk_labels[tmp_label]) {
        tmp_label = blk_labels[tmp_label];
        blk_labels[tid] = tmp_label;
    }

    /// local union find
    if (blk_x_idx != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - 1 + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            _union_find(blk_labels, tid, tid - 1);
    }
    __syncthreads();

    if (blk_y_idx != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - blk_w + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            _union_find(blk_labels, tid, tid - blk_w);
    }
    __syncthreads();

    /// store to global index
    labels[gid] = blk_labels[tid];
}

template<int blk_w, int blk_h, int ch>
__global__ void _boundary_analysis_h(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = (blk_y_idx + threadIdx.y) * blk_h;
    int gid = col * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for (int i = 0; i < ch; i++) {
        diff = tex2D<float>(in_tex, col, row * i) - tex2D<float>(in_tex, col, row * i - 1);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _union_find(labels, gid, gid - width);
    }
}

template<int blk_w, int blk_h, int ch>
__global__ void _boundary_analysis_v(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = (blk_x_idx + threadIdx.x) * blk_w;
    int row = blk_y_idx + threadIdx.y;
    int gid = col * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for (int i = 0; i < ch; i++) {
        diff = tex2D<float>(in_tex, col, row * i) - tex2D<float>(in_tex, col - 1, row * i);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _union_find(labels, gid, gid - 1);
    }
}

template<int blk_w, int blk_h, int ch>
__global__ void _global_path_compression(int *labels, int width, int height) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int gid = col * width + col;
    int label, old_label;

    if (row >= height || col >= width)
        return;

    old_label = labels[gid];
    label = labels[old_label];

    while (old_label != label) {
        old_label = label;
        label = labels[label];
    }
    __syncthreads();

    labels[gid] = label;
}

__global__ void _label_gen_map(int *labels, int *map, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        map[labels[gid]] = gid;
    }
}

__global__ void _label_remap(int *labels, int *map, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        labels[gid] = map[labels[gid]];
    }
}

__global__ void _ms_filter(float3 *img,
                           float3 *dst,
                           int width,
                           int height,
                           int dis_range,
                           float color_range,
                           float min_shift,
                           int max_iter) {

    int tid = (blockDim.x * blockIdx.x + threadIdx.x) +
              (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.y * blockDim.y;
    int i,j;
    while (true) {
        //printf("%d\n",tid);
        j = tid % width;
        i = tid / width;
        //printf("%d %d %d\n", tid, j, i);
        //printf("%d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
        //int j=blockDim.x*blockIdx.x+threadIdx.x;
        //int i=blockDim.y*blockIdx.y+threadIdx.y;
        if (j >= width || i >= height)
            break;
        //printf("%f %f %f",img[i*width+j].x,img[i*width+j].y,img[i*width+j].z);
        int ic = i;
        int jc = j;
        int ic_old, jc_old;
        float L_old, U_old, V_old;
        float L = img[i * width + j].x;
        float U = img[i * width + j].y;
        float V = img[i * width + j].z;
        L = L * 100 / 255;
        U = U - 128;
        V = V - 128;

        double shift = 5;
        int i2_from = max(0, i - dis_range), i2to = min(height, i + dis_range + 1);
        int j2_from = max(0, j - dis_range), j2to = min(width, j + dis_range + 1);
        for (int iters = 0; shift > 3 && iters < 5; iters++) {
            ic_old = ic;
            jc_old = jc;
            L_old = L;
            U_old = U;
            V_old = V;

            float mi = 0;
            float mj = 0;
            float mL = 0;
            float mU = 0;
            float mV = 0;
            int num = 0;

            for (int i2 = i2_from; i2 < i2to; i2++) {
                for (int j2 = j2_from; j2 < j2to; j2++) {
                    float L2 = img[i2 * width + j2].x, U2 = img[i2 * width + j2].y, V2 = img[i2 * width + j2].z;
                    if(i==1&&j==0){
                        printf("J2 i2 L2 U2 V2 %d %d %f %f %f\n",j2,i2,L2,U2,V2);
                    }
                    L2 = L2 * 100 / 255;
                    U2 = U2 - 128;
                    V2 = V2 - 128;
                    double dL = L2 - L;
                    double dU = U2 - U;
                    double dV = V2 - V;
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

            if(i==1&&j==0){
                printf("L U V %f %f %f\n",L,U,V);
                printf("L U V %f %f %f\n",mL,mU,mV);
            }

            L = mL * num_;
            U = mU * num_;
            V = mV * num_;
            ic = lround(mi * num_ + 0.5);
            jc = lround(mj * num_ + 0.5);
            int di = ic - ic_old;
            int dj = jc - jc_old;
            double dL = L - L_old;
            double dU = U - U_old;
            double dV = V - V_old;

            shift = di * di + dj * dj + dL * dL + dU * dU + dV * dV;
        }

        L = L * 255 / 100;
        U = U + 128;
        V = V + 128;
        dst[i * width + j].x = L;
        dst[i * width + j].y = U;
        dst[i * width + j].z = V;
        //printf("%f %")
        if(i==1&&j==0)
            printf("%f %f %f\n", dst[i * width + j].x, dst[i * width + j].y, dst[i * width + j].z);
        tid+=gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
    //printf("Finish\n");
    //int a=threadIdx.x;
    //dst[0].x = 666;
    //printf("test %f\n", img[0].x);
}

namespace CuMeanShift {
    template<int blk_w, int blk_h, int ch, bool ign>
    void CudaUnionFind<blk_w, blk_h, ch, ign>::union_find(float3 *img,
                                                          float3 *dst,
                                                          int width,
                                                          int height,
                                                          int dis_range,
                                                          float color_range,
                                                          float min_shift,
                                                          int max_iter) {
        dim3 grid(16, 16);
        dim3 block(16, 16);
        _ms_filter << < grid, block >> > (img,dst,width,height,dis_range,color_range,min_shift,max_iter);
        float3* host_res=(float3*)malloc(width*height* sizeof(float3));
        cudaMemcpy(host_res, dst, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);
        printf("hell0 %f %f %f\n", host_res[0].x,host_res[0].y,host_res[0].z);
        printf("hell0 %f %f %f\n", host_res[1].x,host_res[1].y,host_res[1].z);
        /// create texture object
        /*
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

        dim3 block_1(blk_w, blk_h);
        dim3 grid_1(CEIL(width, blk_w), CEIL(height, blk_h));
        dim3 grid_2(CEIL(width, blk_w), CEIL(FLOOR(height - 1, blk_h), blk_h));
        dim3 grid_3(CEIL(FLOOR(width - 1, blk_w), blk_w), CEIL(height, blk_h));

        cudaMemcpy(new_labels, labels, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc(&tmp_labels, width * height * sizeof(int));
        cudaMalloc(&labels_map, width * height * sizeof(int));

        _local_union_find<blk_w, blk_h, ch, ign><<<grid_1, block_1>>>(in_tex, new_labels, width, height, range);
        _boundary_analysis_h<blk_w, blk_h, ch><<<grid_2, block_1>>>(in_tex, new_labels, width, height, range);
        _boundary_analysis_v<blk_w, blk_h, ch><<<grid_3, block_1>>>(in_tex, new_labels, width, height, range);
        _global_path_compression<blk_w, blk_h, ch><<<grid_1, block_1>>>(new_labels, width, height);

        /// count labels
        cudaMemcpy(tmp_labels, new_labels, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
        thrust::device_ptr<int> labels_ptr(tmp_labels);
        thrust::device_ptr<int> new_end = thrust::unique(thrust::device, labels_ptr, labels_ptr + width * height);
        *label_count = new_end - labels_ptr;

        /// remap labels
        dim3 block_gen_map(blk_w * blk_h, 1);
        dim3 grid_gen_map(CEIL(*label_count, blk_w * blk_h), 1);
        dim3 block_remap(blk_w * blk_h, 1);
        dim3 grid_remap(CEIL(width * height, blk_w * blk_h), 1);

        _label_gen_map<<<grid_gen_map, block_gen_map>>>(tmp_labels, labels_map, *label_count);
        _label_remap<<<grid_remap, block_remap>>>(new_labels, labels_map, width * height);
        cudaDeviceSynchronize();
        cudaDestroyTextureObject(in_tex);
        cudaFree(tmp_labels);
        cudaFree(labels_map);*/
    }
}