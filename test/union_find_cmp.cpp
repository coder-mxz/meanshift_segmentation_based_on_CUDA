//
// Created by iffi on 19-6-7.
//
#include <CImg.h>
#include <cmath>
#include <stack>
#include <fstream>
#include <iostream>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define color_is_same(a, b) (((a)[0] == (b)[0]) && ((a)[1] == (b)[1]) && ((a)[2] == (b)[2]))

using namespace std;
using namespace cimg_library;
/**
 * @brief: first argument: test binfile path,
 *         second argument: reference binfile path
 */
int main(int argc, char **argv) {
    /// read image
    if (argc != 3) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 2" << endl;
        return 1;
    }
    CImg<uint8_t> img(argv[1]), org_img(img);
    CImg<int> labels(img.width(), img.height(), 1, 1, -1);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    } else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }

    cout << "Regions count:" << union_find(img, labels) << endl;

    CImgDisplay disp(labels, "labels", 1);
    disp.show();
    while (!disp.is_closed()) {
        disp.wait();
    }

    if (!outputBin(argv[2], labels.size(), labels.data())) {
        cout << "Failed to output bin file" << endl;
        return 1;
    }
    return 0;
}
