#include <utils.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_union_find/cuda_union_find.h>
#include <driver_types.h>

/**
 * @brief UnionFind function
 * @param labels: labels array, on shared meory or global memory
 * @param idx_1: index 1
 * @param idx_2: index 2
 */
__device__ void _unionFind(int *labels, int idx_1, int idx_2) {
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

/**
 * @brief Local UnionFind function
 * @tparam blk_w: block width
 * @tparam blk_h: block height
 * @tparam ch: channels
 * @tparam ignore_labels: whether ignore input labels or not
 * @param in_tex: input image/vector matrix
 * @param labels: input labels
 * @param width: input image width
 * @param height: input image height
 * @param range: maximum join threshold of || pixel1 - pixel2 ||
 */
template <int blk_w, int blk_h, int ch, bool ignore_labels=false>
__global__ void _localUnionFind(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    __shared__ int blk_labels[blk_w * blk_h];
    __shared__ int blk_org_labels[blk_w * blk_h];
    __shared__ float blk_pixels[blk_w * blk_h * ch];

    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int blk_size = blk_w * blk_h;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int tid = threadIdx.y * blk_w + threadIdx.x;
    int gid = row * width + col;

    if (row >= height || col >= width)
        return;

    /// read labels
    blk_labels[tid] = tid;
    if (ignore_labels) {
        blk_org_labels[tid] = gid;
    }
    else {
        blk_org_labels[tid] = labels[gid];
    }

    /// read pixels
    for(int i=0; i<ch; i++) {
        blk_pixels[tid + blk_size * i] = tex2D<float>(in_tex, col, row + height * i);
    }

    __syncthreads();

    /// row scan
    float diff, diff_sum = 0;

    if (threadIdx.x != 0) {
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
    if (threadIdx.y != 0) {
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
    }
    blk_labels[tid] = tmp_label;

    /// local union find
    diff_sum = 0;
    if (threadIdx.x != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - 1 + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            _unionFind(blk_labels, tid, tid - 1);
    }
    __syncthreads();

    diff_sum = 0;
    if (threadIdx.y != 0) {
        for (int i = 0; i < ch; i++) {
            diff = blk_pixels[tid + i * blk_size] - blk_pixels[tid - blk_w + i * blk_size];
            diff_sum += diff * diff;
        }
        if (diff_sum < range)
            _unionFind(blk_labels, tid, tid - blk_w);
    }
    __syncthreads();

    /// store to global index
    labels[gid] = blk_org_labels[blk_labels[tid]];
}

/**
 * @brief horizontal boundary analysis
 * @tparam blk_w: block width
 * @tparam blk_h: block height
 * @tparam ch: channels
 * @param in_tex: input image/vector matrix
 * @param labels: input labels
 * @param width: input image width
 * @param height: input image height
 * @param range: maximum join threshold of || pixel1 - pixel2 ||
 */
template <int blk_w, int blk_h, int ch>
__global__ void _boundaryAnalysisH(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = (blk_y_idx + threadIdx.y + 1) * blk_h;
    int gid = row * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for(int i=0; i<ch; i++) {
        diff = tex2D<float>(in_tex, col, row + height * i) - tex2D<float>(in_tex, col, row - 1 + height * i);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _unionFind(labels, gid, gid - width);
    }
}

/**
 * @brief vertical boundary analysis
 * @tparam blk_w: block width
 * @tparam blk_h: block height
 * @tparam ch: channels
 * @param in_tex: input image/vector matrix
 * @param labels: input labels
 * @param width: input image width
 * @param height: input image height
 * @param range: maximum join threshold of || pixel1 - pixel2 ||
 */
template <int blk_w, int blk_h, int ch>
__global__ void _boundaryAnalysisV(cudaTextureObject_t in_tex, int *labels, int width, int height, float range) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = (blk_x_idx + threadIdx.x + 1) * blk_w;
    int row = blk_y_idx + threadIdx.y;
    int gid = row * width + col;

    if (row >= height || col >= width)
        return;

    float diff, diff_sum = 0;
    for(int i=0; i<ch; i++) {
        diff = tex2D<float>(in_tex, col, row + height * i) - tex2D<float>(in_tex, col - 1, row + height * i);
        diff_sum += diff * diff;
    }
    if (diff_sum < range) {
        _unionFind(labels, gid, gid - 1);
    }
}

/**
 * @brief global path compression
 * @param labels: input labels
 * @param width: input image width(labels width)
 * @param height: input image height(labels height)
 */
__global__ void _globalPathCompression(int *labels, int width, int height) {
    int blk_x_idx = blockIdx.x * blockDim.x;
    int blk_y_idx = blockIdx.y * blockDim.y;
    int col = blk_x_idx + threadIdx.x;
    int row = blk_y_idx + threadIdx.y;
    int gid = row * width + col;
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

/**
 * @brief generate label map from labels
 * @param labels: input labels
 * @param map: empty map array, all elemets must be set to 0
 * @param count: size of labels array
 */
__global__ void _labelGenMap(int *labels, int *map, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        map[labels[gid]] = 1;
    }
}

/**
 * @brief generate label index from label map, index is from 1 to n,
 *        n is the number of total different labels.
 * @param map: input map
 * @param counter: global counter
 * @param count: size of labels array
 */
__global__ void _labelGenIndex(int *map, int *counter, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        if (map[gid] > 0)
            map[gid] = atomicAdd(counter, 1);
        else
            map[gid] = -1;
    }
}

/**
 * @brief remap labels
 * @param labels: input labels
 * @param map: input map
 * @param count: size of labels array
 */
__global__ void _labelRemap(int *labels, int *map, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        labels[gid] = map[labels[gid]];
    }
}

namespace CuMeanShift {
    template <int blk_w, int blk_h, int ch, bool ign>
    void CudaUnionFind<blk_w, blk_h, ch, ign>::unionFind(int *labels,
                                                          float *input,
                                                          int *new_labels,
                                                          int *label_count,
                                                          int pitch,
                                                          int width,
                                                          int height,
                                                          float range) {
        int *labels_map;

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

        if (!ign)
            cudaMemcpy(new_labels, labels, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
        _localUnionFind<blk_w, blk_h, ch, ign><<<grid_1, block_1>>>(in_tex, new_labels, width, height, range);

        if (height > blk_h)
            _boundaryAnalysisH<blk_w, blk_h, ch><<<grid_2, block_1>>>(in_tex, new_labels, width, height, range);
        if (width > blk_w)
            _boundaryAnalysisV<blk_w, blk_h, ch><<<grid_3, block_1>>>(in_tex, new_labels, width, height, range);

        _globalPathCompression<<<grid_1, block_1>>>(new_labels, width, height);

        cudaDeviceSynchronize();
        /// count labels & remap labels
        int *counter_dev;
        dim3 block_map(blk_w * blk_h, 1);
        dim3 grid_map(CEIL(width * height, blk_w * blk_h), 1);

        cudaMalloc(&counter_dev, sizeof(int));
        cudaMemset(counter_dev, 0, sizeof(int));
        cudaMalloc(&labels_map, width * height * sizeof(int));
        cudaMemset(labels_map, 0, width * height * sizeof(int));

        _labelGenMap<<<grid_map, block_map>>>(new_labels, labels_map, width * height);
        _labelGenIndex<<<grid_map, block_map>>>(labels_map, counter_dev, width * height);
        _labelRemap<<<grid_map, block_map>>>(new_labels, labels_map, width * height);

        cudaMemcpy(label_count, counter_dev, sizeof(int), cudaMemcpyDeviceToHost);


        cudaDeviceSynchronize();
        cudaDestroyTextureObject(in_tex);
        cudaFree(counter_dev);
        cudaFree(labels_map);
    }
}