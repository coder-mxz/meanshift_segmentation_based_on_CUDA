//
// Created by iffi on 19-6-7.
//
#include <cmath>
#include <stack>
#include <fstream>
#include <iostream>

using namespace std;

bool readBin(const char *path, float **data, size_t *count) {
    ifstream file(path, ios_base::in | ios_base::binary);
    if (!file.is_open())
        return false;
    file.read((char*) count, sizeof(size_t));
    *data = new float[*count * sizeof(float)];
    file.read((char*) *data, *count * sizeof(float));
    file.close();
    return true;
}

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

    float *test, *ref;
    size_t test_size, ref_size;
    size_t diff_grade[5] = {0};

    if (!readBin(argv[1], &test, &test_size)) {
        cout << "Failed to read test file" << endl;
        return 1;
    }
    if (!readBin(argv[2], &ref, &ref_size)) {
        cout << "Failed to read ref file" << endl;
        return 1;
    }

    if (test_size != ref_size) {
        cout << "Unmatched data count, test file: " << test_size << ", ref file: " << ref_size << endl;
        return 2;
    }

    for(size_t i=0; i<ref_size; i++) {
        float diff_ratio = abs(test[i] - ref[i]) / ref[i];
        if (diff_ratio >= 0.1) {
            diff_grade[0]++;
        }
        else if(diff_ratio >= 0.05) {
            diff_grade[1]++;
        }
        else if(diff_ratio >= 0.01) {
            diff_grade[2]++;
        }
        else if(diff_ratio >= 0.005) {
            diff_grade[3]++;
        }
        else
            diff_grade[4]++;
    }

    delete [] test;
    delete [] ref;

    cout << "difference ratio:" << endl;
    cout << ">= 10%:  " << diff_grade[0] * 1.0 / ref_size << "(" << diff_grade[0] << ")" << endl;
    cout << ">= 5%:   " << diff_grade[1] * 1.0 / ref_size << "(" << diff_grade[0] << ")" << endl;
    cout << ">= 1%:   " << diff_grade[2] * 1.0 / ref_size << "(" << diff_grade[0] << ")" << endl;
    cout << ">= 0.5%: " << diff_grade[3] * 1.0 / ref_size << "(" << diff_grade[0] << ")" << endl;
    cout << "< 0.5%:  " << diff_grade[4] * 1.0 / ref_size << "(" << diff_grade[0] << ")" << endl;
    return 0;
}
