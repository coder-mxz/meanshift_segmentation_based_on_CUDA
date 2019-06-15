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
    file.write((const char *) &count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}

template <int ch = 1, int rad = 4>
void flooding(CImg<float> &img, int *output, int loop, float color_range) {
    // initial labels
    for (int row = 0; row < img.height(); row++) {
        for (int col = 0; col < img.width(); col++) {
            output[row * img.width() + col] = (row * img.width() + col) * 10;
        }
    }
    // process LOOP_TIMES times
    for (int _i = 0; _i < loop; _i++) {
        for (int row = 0; row < img.height(); row++) {
            for (int col = 0; col < img.width(); col++) {
                // process every pixel
                float min_delta_luv = 999999.0f;
                int min_row = 999999, min_col = 999999;
                for (int r = -rad; r < 0; r++) {
                    int cur_row = row + r, cur_col = col + r;
                    // find up
                    if (cur_row >= 0) {
                        float delta_luv = 0.0f;
                        for (int chn = 0; chn < ch; chn++) {
                            float delta =
                                    img.atXY(col, cur_row, 0, chn) - img.atXY(col, row, 0, chn);
                            delta_luv += delta * delta;
                        }
                        if (delta_luv < min_delta_luv) {
                            min_delta_luv = delta_luv;
                            min_row = cur_row;
                            min_col = col;
                        }
                    }
                    // find left
                    if (cur_col >= 0) {
                        float delta_luv = 0.0f;
                        for (int chn = 0; chn < ch; chn++) {
                            float delta =
                                    img.atXY(cur_col, row, 0, chn) - img.atXY(col, row, 0, chn);
                            delta_luv += delta * delta;
                        }
                        if (delta_luv < min_delta_luv) {
                            min_delta_luv = delta_luv;
                            min_row = row;
                            min_col = cur_col;
                        }
                    }
                }
                // set label
                if (min_delta_luv < color_range) {
                    output[row * img.width() + col] =
                            output[min_row * img.width() + min_col];
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int loops = strtol(argv[1], nullptr, 10);
    CudaFlooding<32, 32, 1, 4> f;

    /// generate test image
    CImg<float> img(WIDTH, HEIGHT, 1, 1, 0.0f);
    CImg<int> cpu_labels, gpu_labels;
    srand(0);
    for (int i = 0; i < SQUARE_COUNT; i++) {
        int x = rand() % WIDTH, y = rand() % HEIGHT;
        for (int xx = max(x - 10, 0); xx < min(x + 10, WIDTH - 1); xx++) {
            for (int yy = max(y - 10, 0); yy < min(y + 10, HEIGHT - 1); yy++) {
                img.atXY(xx, yy, 0, 0) = (i + 1) * 10.0f;
            }
        }
    }


    /******************
     * CPU version
     ******************/

    // initialize label array
    int *cpu_output = new int[WIDTH * HEIGHT];
    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            cpu_output[y * WIDTH + x] = (y * WIDTH + x) * 10;
        }
    }

    flooding(img, cpu_output, loops, 1.0f);

    /******************
     * GPU version
     ******************/
    size_t pitch;
    float *dev_image;
    int *dev_output, *host_output;
    host_output = new int[WIDTH * HEIGHT];

    cudaMalloc(&dev_output, sizeof(int) * WIDTH * HEIGHT);
    cudaMallocPitch(
            &dev_image,
            &pitch,
            WIDTH * sizeof(float),
            HEIGHT);
    cudaMemcpy2D(
            dev_image,
            pitch,
            img.data(),
            WIDTH * sizeof(float),
            WIDTH * sizeof(float),
            HEIGHT,
            cudaMemcpyHostToDevice);

    f.flooding(dev_output, dev_image, pitch, WIDTH, HEIGHT, loops, 1.0f);
    cudaMemcpy(host_output, dev_output, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    // show image
    cpu_labels.assign(cpu_output, WIDTH, HEIGHT, 1, 1, true);
    gpu_labels.assign(host_output, WIDTH, HEIGHT, 1, 1, true);

    CImgDisplay disp1(img, "original image", 1);
    CImgDisplay disp2(cpu_labels, "CPU labels", 1);
    CImgDisplay disp3(gpu_labels, "GPU labels", 1);
    disp1.show();
    disp2.show();
    disp3.show();
    while (!disp1.is_closed() || !disp2.is_closed() || !disp3.is_closed()) {
        disp1.wait();
        disp2.wait();
        disp3.wait();
    }

    cudaFree(dev_image);
    cudaFree(dev_output);
    delete [] cpu_output;
    delete [] host_output;
}
