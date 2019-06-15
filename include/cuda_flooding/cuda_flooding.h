#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template<int blk_width = 32, int blk_height = 32, int channels = 1, int radius = 4>
    class CudaFlooding {
    public:
        /**
         * @fn flooding
         * @brief flooding algorithm
         * @param labels: Labels array on device memory
         * @param input: Input image on device memory, aligned by pitch
         * @param color_range: Maximum LUV threshold used to join pixels
         */
        void flooding(int *labels,
                      float *input,
                      int pitch,
                      int width,
                      int height,
                      int loops,
                      float color_range);
    };
}