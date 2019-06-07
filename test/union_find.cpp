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

struct Point {
    int x, y;

    Point(int x, int y) : x(x), y(y) {}
};

class RAList {
public:
    int label;
    RAList *next;

    RAList() : label(-1), cur(nullptr), next(nullptr) {}

    ~RAList() = default;

    bool Insert(RAList *entry) {
        if (next == nullptr) {
            next = entry;
            entry->next = nullptr;
            return false;
        } else if (next->label > entry->label) {
            entry->next = next;
            next = entry;
            return false;
        }
        bool exists = false;
        cur = next;
        while (cur != nullptr) {
            if (entry->label == cur->label) {
                exists = true;
                break;
            } else if ((!(cur->next)) || (cur->next->label > entry->label)) {
                entry->next = cur->next;
                cur->next = entry;
                break;
            }
            cur = cur->next;
        }
        return exists;
    }

private:
    RAList *cur, *prev;
};

bool outputBin(const char *path, size_t count, int *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}

int union_find(CImg<uint8_t> &img, CImg<int> &labels) {
    int region_count = 0;
    uint8_t *mode = new uint8_t[img.height() * img.width() * 3];

    int label = -1;
    for (int i = 0; i < img.height(); i++) {
        for (int j = 0; j < img.width(); j++) {
            labels(j, i) = ++label;

            mode[label * 3 + 0] = img(j, i, 0);
            mode[label * 3 + 1] = img(j, i, 1);
            mode[label * 3 + 2] = img(j, i, 2);

            // Fill
            std::stack<Point> neigh_stack;
            neigh_stack.push(Point(i, j));
            const int dxdy[][2] = {{-1, -1},
                                   {-1, 0},
                                   {-1, 1},
                                   {0,  -1},
                                   {0,  1},
                                   {1,  -1},
                                   {1,  0},
                                   {1,  1}};
            while (!neigh_stack.empty()) {
                Point p = neigh_stack.top();
                neigh_stack.pop();
                for (int k = 0; k < 8; k++) {
                    int i2 = p.x + dxdy[k][0], j2 = p.y + dxdy[k][1];
                    if (i2 >= 0 && j2 >= 0 && i2 < img.height() && j2 < img.width() && labels(j2, i2) < 0 &&
                        img(j, i) == img(j2, i2)) {
                        labels(j2, i2) = label;
                        neigh_stack.push(Point(i2, j2));
                    }
                }
            }

        }
    }
    //current Region count
    region_count = label + 1;

    int old_region_count = region_count;


    // TransitiveClosure
    RAList *ra_list = new RAList[region_count], *ra_pool = new RAList[img.height() * img.width() * 10];
    int *label_buffer = new int[region_count];

    for (int counter = 0, delta_region_count = 1; counter < 5 && delta_region_count > 0; counter++) {
        // 1.Build RAM using classifiction structure

        for (int i = 0; i < region_count; i++) {
            ra_list[i].label = i;
            ra_list[i].next = nullptr;
        }
        for (int i = 0; i < region_count * 10 - 1; i++) {
            ra_pool[i].next = &ra_pool[i + 1];
        }
        ra_pool[region_count * 10 - 1].next = nullptr;
        RAList *ra_node1, *ra_node2, *old_free_list, *free_list = ra_pool;
        for (int i = 0; i < img.height(); i++)
            for (int j = 0; j < img.width(); j++) {
                if (i > 0 && labels(j, i) != labels(j, i - 1)) {
                    // Get 2 free node
                    ra_node1 = free_list;
                    ra_node2 = free_list->next;
                    old_free_list = free_list;
                    free_list = free_list->next->next;
                    // connect the two region
                    ra_node1->label = labels(j, i);
                    ra_node2->label = labels(j, i - 1);
                    if (ra_list[labels(j, i)].Insert(ra_node2))    //already exists!
                        free_list = old_free_list;
                    else
                        ra_list[labels(j, i - 1)].Insert(ra_node1);
                }
                if (j > 0 && labels(j, i) != labels(j - 1, i)) {
                    // Get 2 free node
                    ra_node1 = free_list;
                    ra_node2 = free_list->next;
                    old_free_list = free_list;
                    free_list = free_list->next->next;
                    // connect the two region
                    ra_node1->label = labels(j, i);
                    ra_node2->label = labels(j - 1, i);
                    if (ra_list[labels(j, i)].Insert(ra_node2))
                        free_list = old_free_list;
                    else
                        ra_list[labels(j - 1, i)].Insert(ra_node1);
                }
            }

        // 2.Treat each region Ri as a disjoint set
        for (int i = 0; i < region_count; i++) {
            RAList *neighbor = ra_list[i].next;
            while (neighbor) {
                if (color_is_same(&mode[3 * i], &mode[3 * neighbor->label])) {
                    int i_can_el = i, neigh_can_el = neighbor->label;
                    while (ra_list[i_can_el].label != i_can_el)
                        i_can_el = ra_list[i_can_el].label;
                    while (ra_list[neigh_can_el].label != neigh_can_el)
                        neigh_can_el = ra_list[neigh_can_el].label;
                    if (i_can_el < neigh_can_el)
                        ra_list[neigh_can_el].label = i_can_el;
                    else {
                        ra_list[i_can_el].label = neigh_can_el;
                    }
                }
                neighbor = neighbor->next;
            }
        }
        // 3. Union Find
        for (int i = 0; i < region_count; i++) {
            int i_can_el = i;
            while (ra_list[i_can_el].label != i_can_el) i_can_el = ra_list[i_can_el].label;
            ra_list[i].label = i_can_el;
        }
        // 4. Traverse joint sets, relabeling image.

        label = -1;
        for (int i = 0; i < region_count; i++) {
            label_buffer[i] = -1;
        }
        for (int i = 0; i < region_count; i++) {
            int i_can_el = ra_list[i].label;
            if (label_buffer[i_can_el] < 0) {
                label_buffer[i_can_el] = ++label;
                for (int k = 0; k < 3; k++)
                    mode[label * 3 + k] = mode[i * 3 + k];
            }
        }
        region_count = label + 1;

        for (int i = 0; i < img.height(); i++)
            for (int j = 0; j < img.width(); j++)
                labels(j, i) = label_buffer[ra_list[labels(j, i)].label];

        delta_region_count = old_region_count - region_count;
        old_region_count = region_count;
    }
    //Destroy RAM
    delete[] ra_list;
    delete[] ra_pool;
    delete[] label_buffer;
    return region_count;
}

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

