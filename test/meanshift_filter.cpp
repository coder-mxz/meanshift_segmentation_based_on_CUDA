//
// Created by iffi on 19-6-6.
//
#include <CImg.h>
#include <cmath>
#include <fstream>
#include <iostream>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

using namespace std;
using namespace cimg_library;

bool outputBin(const char *path, size_t count, float *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char*)data, count * sizeof(float));
    file.close();
    return true;
}

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 *         third argument: spatial radius (eg: 13)
 *         forth argument: color radius (eg: 10)
 */
int main(int argc, char **argv) {
    /// read image
    if (argc != 5) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 4" << endl;
        return 1;
    }
    CImg<float> img(argv[1]), org_img(img);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    }
    else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }

    long spatial_radius = strtol(argv[3], nullptr, 10);
    float color_radius = strtof(argv[4], nullptr);

    /// perform meanshift filtering
    /// Note: in cimg, x(j) is column and y(i) is row
    cimg_forXY(img, j, i) {
        /// calculate L,U,V value for each pixel
        int ic = i;
        int jc = j;
        int ic_old, jc_old;
        float L_old, U_old, V_old;
        float L = img(j, i, 0);
        float U = img(j, i, 1);
        float V = img(j, i, 2);
        L = L * 100 / 255;
        U = U - 128;
        V = V - 128;

        double shift = 5;
        int i2_from = max(0, i - spatial_radius), i2to = min(img.height(), i + spatial_radius + 1);
        int j2_from = max(0, j - spatial_radius), j2to = min(img.width(), j + spatial_radius + 1);
        for (int iters = 0; shift > 3 && iters < 5; iters++) {
            ic_old = ic;
            jc_old = jc;
            L_old = L;
            U_old = U;
            V_old = V;

            float mi = 0;
            float mj = 0;
            float mL = 0;
            float mU = 0;
            float mV = 0;
            int num = 0;

            for (int i2 = i2_from; i2 < i2to; i2++) {
                for (int j2 = j2_from; j2 < j2to; j2++) {
                    float L2 = img(j2, i2, 0), U2 = img(j2, i2, 1), V2 = img(j2, i2, 2);
                    L2 = L2 * 100 / 255;
                    U2 = U2 - 128;
                    V2 = V2 - 128;

                    double dL = L2 - L;
                    double dU = U2 - U;
                    double dV = V2 - V;
                    if (dL * dL + dU * dU + dV * dV <= color_radius) {
                        mi += i2;
                        mj += j2;
                        mL += L2;
                        mU += U2;
                        mV += V2;
                        num++;
                    }
                }
            }
            float num_ = 1.f / num;
            L = mL * num_;
            U = mU * num_;
            V = mV * num_;
            ic = lround(mi * num_ + 0.5);
            jc = lround(mj * num_ + 0.5);
            int di = ic - ic_old;
            int dj = jc - jc_old;
            double dL = L - L_old;
            double dU = U - U_old;
            double dV = V - V_old;

            shift = di * di + dj * dj + dL * dL + dU * dU + dV * dV;
        }

        L = L * 255 / 100;
        U = U + 128;
        V = V + 128;
        img(j, i, 0) = L;
        img(j, i, 1) = U;
        img(j, i, 2) = V;
    }

    CImgDisplay disp(img, "filtered", 0);
    CImgDisplay disp2(org_img, "original", 0);
    disp.show();
    disp2.show();
    while (!disp.is_closed() && !disp2.is_closed()) {
        disp.wait();
        disp2.wait();
    }
    if (!outputBin(argv[2], img.size(), img.data())) {
        cout << "Failed to output bin file" << endl;
        return 1;
    }
    return 0;
}