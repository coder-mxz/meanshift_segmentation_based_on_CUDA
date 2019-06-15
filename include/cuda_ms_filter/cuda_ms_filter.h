#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template <int blk_width=32, int blk_height=32, int channels=1>
    class CudaMsFilter {
    public:
        void msFilterLUV(float *input,
                           float *output,
                           int width,
                           int height,
                           int pitch,
                           int dis_range,
                           float color_range);
    };
}