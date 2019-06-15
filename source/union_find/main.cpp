//
// Created by mxz on 2019/6/15.
//

#include <CImg.h>
#include <cmath>
#include <stack>
#include <fstream>
#include<pthread.h>
#include<cstdlib>
#include<cstring>
#include<map>
#include<iostream>
#include<time.h>


#include "union_find.cpp"

using namespace std;
using namespace cimg_library;

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 */
int main(int argc, char **argv) {
    /// read image
    if (argc != 3) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 2" << endl;
        return 1;
    }
    CImg <uint8_t> img(argv[1]), org_img(img);
    CImg<int> labels(img.width(), img.height(), 1, 1, -1);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    } else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }
/**        CImgDisplay disp(img, "img", 1);
        disp.show();
        while (!disp.is_closed()) {
            disp.wait();
        }
**/

    clock_t start,finish;
    double totaltime;

    start=clock();
    PthreadUnionFind puf1(img, labels);
    puf1.unionFindOneThread();
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"\n one_thread runtime of file "<<argv[1]<<" is "<<totaltime<<"second"<<endl;

    start=clock();
    PthreadUnionFind puf2(img, labels);
    puf2.unionFindMultiThread();
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"\n multi_thread runtime of file "<<argv[1]<<" is "<<totaltime<<"second"<<endl;

    outputBin(argv[2],labels.size(),puf2.getResultLabels());
}