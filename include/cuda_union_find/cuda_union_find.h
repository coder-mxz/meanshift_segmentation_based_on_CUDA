#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
    template <int blk_width=32, int blk_height=32, int channels=1, bool ignore_labels=false>
    class CudaUnionFind {
    public:
        void union_find(int *labels,
                        float *input,
                        int *new_labels,
                        int *label_count,
                        int pitch,
                        int width,
                        int height,
                        float range);
    };
}