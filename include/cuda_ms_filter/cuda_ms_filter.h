/**
 * @file cuda_ms_filter.h
 * @author xjz and iffi
 * @date 2019-6-15
 */

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template <int blk_width=32, int blk_height=32, int channels=1>
    class CudaMsFilter {
    public:
        /**
         * @brief meanshift filter algorithm
         * @param input: input image on device memory, aligned by pitch
         * @param output: output image array on device memory, aligned by pitch
         * @param width: input image width
         * @param height: input image height
         * @param pitch: input image pitch
         * @param dis_range: scan distance range
         * @param color_range: maximum include threshold of || pixel1 - pixel2 ||
         */
        void msFilterLUV(float *input,
                           float *output,
                           int width,
                           int height,
                           int pitch,
                           int dis_range,
                           float color_range);
    };
}