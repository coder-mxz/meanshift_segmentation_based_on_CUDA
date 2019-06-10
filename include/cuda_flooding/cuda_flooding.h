#include <CImg.h>
using namespace cimg_library;

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#ifdef __JETBRAINS_IDE__
#include <cuda_fake_headers.h>
#endif

namespace CuMeanShift {
template <int blk_width = 32, int blk_height = 32, int channels = 1,
          int radius = 4>
class CudaFlooding {
public:
  /*
   * @fn flooding
   * @param in_tex 二维数组，具体定义为texture<float, 2,
   * cudaReadModeElementType>，具体内容跟“data_3: 每行 w * sizeof(float), 共h *
   * x行”定义一样
   * @param output cudaMalloc申请的device内存，大小为width*height*sizeof(int)
   * @param width 图像宽度
   * @param height 图像高度
   * @param color_range 当作相同标记时的$\deltaLUV$范围
   * @brief: 对图像进行flooding
   */

  void flooding(cudaTextureObject_t in_tex, int *output, int width, int height,
                float color_range);

  void _test_flooding(CImg<float> &img, int *output, float color_range);
};
}