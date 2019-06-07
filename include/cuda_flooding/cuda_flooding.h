#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

texture<float, 2, cudaReadModeElementType> in_tex; //存储图像输入


template <int radius = 4>
__global__ void flooding(int *output, int channels, int pitch, int width,
                         int height, int row_offset, int col_offset,
                         float color_range);
