#include <CImg.h>
#include <cstdlib>
#include <cstring>
#include <cuda_flooding/cuda_flooding.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace CuMeanShift;
using namespace cimg_library;

const int SQUARE_COUNT = 10;
const int WIDTH = 640;
const int HEIGHT = 480;

bool outputBin(const char *path, size_t count, int *data) {
  ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
  if (!file.is_open())
    return false;
  file.write((const char *)&count, sizeof(size_t));
  file.write((const char *)data, count * sizeof(int));
  file.close();
  return true;
}

void displayImage(CImgDisplay &disp) {
  disp.show();
  while (!disp.is_closed()) {
    disp.wait();
  }
}

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 *         third argument: color range
 */
int main(int argc, char *argv[]) {
  CudaFlooding<32, 32, 1, 4> f;
  /// generate test image
  CImg<float> img(WIDTH, HEIGHT, 1, 1, 0.0f);
  srand(0);
  for (int i = 0; i < SQUARE_COUNT; i++) {
    int x = rand() % WIDTH, y = rand() % HEIGHT;
    for (int xx = max(x - 10, 0); xx < min(x + 10, WIDTH - 1); xx++) {
      for (int yy = max(y - 10, 0); yy < min(y + 10, HEIGHT - 1); yy++) {
        img.atXY(xx, yy, 0, 0) = (i + 1) * 10.0f;
      }
    }
  }
  CImg<float> orig_img(img);
  CImg<int> labels(img.width(), img.height(), 1, 1, -1);

  // process image
  int *output = new int[WIDTH * HEIGHT];
  memset(output, 0, sizeof(int) * WIDTH * HEIGHT);
  f._test_flooding(img, output, 1.0f);

  // show image
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      labels.atXY(x, y, 0, 0) = output[y * WIDTH + x];
    }
  }

  CImgDisplay disp1(orig_img, "original image", 1);
  displayImage(disp1);

  CImgDisplay disp2(labels, "labels", 1);
  displayImage(disp2);
}
