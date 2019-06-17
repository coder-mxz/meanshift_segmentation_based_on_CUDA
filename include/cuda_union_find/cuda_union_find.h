/**
 * @file cuda_union_find.h
 * @author iffi
 * @date 2019-6-15
 */

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template <int blk_width=32, int blk_height=32, int channels=1, bool ignore_labels=false>
    class CudaUnionFind {
    public:
        /**
         * @brief Union find algorithm
         * @param labels: preset input labels, optional, could be nullptr if ignore_labels=true
         * @param input: input image
         * @param new_labels: output labels
         * @param label_count: number of different labels
         * @param pitch: input image pitch
         * @param width: input image width
         * @param height: input image height
         * @param range: maximum join threshold of || pixel1 - pixel2 ||
         */
        void unionFind(int *labels,
                        float *input,
                        int *new_labels,
                        int *label_count,
                        int pitch,
                        int width,
                        int height,
                        float range);
    };
}