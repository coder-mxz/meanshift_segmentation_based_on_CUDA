//
// Created by iffi on 19-6-8.
//
#define cimg_use_opencv
#include <CImg.h>
#include <iostream>
using namespace std;
using namespace cimg_library;

int main(int argc, char **argv) {
    CImg<float> img;
    CImgDisplay display;

    img.load_camera(0, 640, 480, 0, false);
    if ( img.is_empty() ) {
        cerr << "Could not capture from camera 0" << endl;
        return 1;
    }

    display.show();
    while(true) {
        // Load image
        img.load_camera(0, 640, 480, 0, false);
        display.display(img).set_title("captured");
        if ( img.is_empty() || display.is_keyQ() || display.is_keyESC() || display.is_closed())
            break;

        // Do what you want with img

    }

    // Release camera
    img.load_camera(0, 640, 480, 0, true);
    
}
