#include <utils.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_union_find/cuda_union_find.h>
#include <driver_types.h>

__device__ void _union_find(int *labels, int idx_1, int idx_2) {
    while(1) {
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

template <int blk_w, int blk_h, int ch, bool ignore_labels=false>
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
    for(int i=0; i<ch; i++)
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

template <int blk_w, int blk_h, int ch>
__global__ void _boundary_analysis_h(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = (blk_y_idx + threadIdx.y) * blk_h;
    int gid = col * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for(int i=0; i<ch; i++) {
        diff = tex2D<float>(in_tex, col, row * i) - tex2D<float>(in_tex, col, row * i - 1);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _union_find(labels, gid, gid - width);
    }
}

template <int blk_w, int blk_h, int ch>
__global__ void _boundary_analysis_v(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = (blk_x_idx + threadIdx.x) * blk_w;
    int row = blk_y_idx + threadIdx.y;
    int gid = col * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for(int i=0; i<ch; i++) {
        diff = tex2D<float>(in_tex, col, row * i) - tex2D<float>(in_tex, col - 1, row * i);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _union_find(labels, gid, gid - 1);
    }
}

template <int blk_w, int blk_h, int ch>
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

    while(old_label != label) {
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

namespace CuMeanShift {
    template <int blk_w, int blk_h, int ch, bool ign>
    void CudaUnionFind<blk_w, blk_h, ch, ign>::union_find(int *labels,
                                                          float *input,
                                                          int *new_labels,
                                                          int *label_count,
                                                          int pitch,
                                                          int width,
                                                          int height,
                                                          float range) {
        int *tmp_labels, *labels_map;

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
        cudaFree(labels_map);
    }
}