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
   * @param img CImg<float>引用
   * @param output host分配的内存，大小为width*height*sizeof(int)
   * @param color_range 当作相同标记时的$\deltaLUV$范围
   * @brief: 对图像进行flooding
   */
  void flooding(CImg<float> &img, int *output, float color_range);

  void _test_flooding(CImg<float> &img, int *output, float color_range);
};
}