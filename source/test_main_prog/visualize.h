//
// Created by iffi on 19-6-14.
//

#ifndef CUDA_VISUALIZE_H
#define CUDA_VISUALIZE_H

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template <int blk_width=32, int blk_height=32>
    class CudaColorLabels {
    public:
        void color_labels(int *labels,
                          float *image,
                          int label_count,
                          int pitch,
                          int width,
                          int height);
    };
}

#endif //CUDA_VISUALIZE_H
