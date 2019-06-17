/**
 * @file union_find.cpp
 * @brief 多线程实现union_find的并行
 * @author mxz
 * @date 2019-6-15
 */

#define HAVE_STRUCT_TIMESPEC
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
#include <union_find/union_find.h>


#define THREAD_NUM 8 ///< 线程数目
const double color_radius = 6.5;


using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::ofstream;
using namespace cimg_library;


void *threadInsert(void *arg);

namespace CuMeanShift {


    /**
     * @brief 构造函数
     *
     * @param img
     * @param cimg_labels
     * @return None
     */
    PthreadUnionFind::PthreadUnionFind(CImg<uint8_t> &img, CImg<int> &cimg_labels) {
        region_count = 0;
        width = img.width();
        height = img.height();
        labels = new int[width * height];
        new_labels = new int[width * height];
        mode_point_count = new int[height * width];
        mode = new float[img.height() * img.width() * 3];

        memset(mode_point_count, 0, width * height * sizeof(int));
        memset(new_labels, 0, width * height * sizeof(int));

        int label = -1;
        for (int i = 0; i < img.height(); i++) {
            for (int j = 0; j < img.width(); j++) {
                labels[dimTran(i, j)] = ++label;
                mode_point_count[label] += 1;
                mode[label * 3 + 0] = img(j, i, 0);
                mode[label * 3 + 1] = img(j, i, 1);
                mode[label * 3 + 2] = img(j, i, 2);

                /// Fill
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
                        if (i2 >= 0 && j2 >= 0 && i2 < img.height() && j2 < img.width() &&
                            labels[dimTran(i2, j2)] < 0 &&
                            img(j, i) == img(j2, i2)) {
                            labels[dimTran(i2, j2)] = label;
                            mode_point_count[label] += 1;
                            neigh_stack.push(Point(i2, j2));
                        }
                    }
                }

            }
        }
        ///current Region count
        region_count = label + 1;
    }

    /**
     * @brief 单线程实现union_find
     * @return labels会被更新
     */
    void PthreadUnionFind::unionFindOneThread() {
        int old_region_count = region_count;
        for (int counter = 0, delta_region_count = 1; counter < 5 && delta_region_count > 0; counter++) {
            int temp_count = 0;
            // 1.Build RAM using classifiction structure
            RAList *ralist = new RAList[region_count], *raPool = new RAList[10 * region_count];    //10 is hard coded!
            for (int i = 0; i < region_count; i++) {
                ralist[i].label = i;
                ralist[i].next = NULL;
            }
            for (int i = 0; i < region_count * 10 - 1; i++) {
                raPool[i].next = &raPool[i + 1];
            }
            raPool[10 * region_count - 1].next = NULL;
            RAList *ra_node1, *ra_node2, *old_ra_freelist, *free_ra_list = raPool;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) {
                    if (i > 0 && labels[dimTran(i, j)] != labels[dimTran(i - 1, j)]) {
                        // Get 2 free node
                        ra_node1 = free_ra_list;
                        ra_node2 = free_ra_list->next;
                        old_ra_freelist = free_ra_list;
                        free_ra_list = free_ra_list->next->next;
                        temp_count += 2;
                        // connect the two region
                        ra_node1->label = labels[dimTran(i, j)];
                        ra_node2->label = labels[dimTran(i - 1, j)];
                        if (ralist[labels[dimTran(i, j)]].insert(ra_node2))    //already exists!
                        {
                            free_ra_list = old_ra_freelist;
                            temp_count -= 2;
                        } else
                            ralist[labels[dimTran(i - 1, j)]].insert(ra_node1);
                    }
                    if (j > 0 && labels[dimTran(i, j)] != labels[dimTran(i, j - 1)]) {
                        // Get 2 free node
                        ra_node1 = free_ra_list;
                        ra_node2 = free_ra_list->next;
                        old_ra_freelist = free_ra_list;
                        free_ra_list = free_ra_list->next->next;
                        temp_count += 2;
                        // connect the two region
                        ra_node1->label = labels[dimTran(i, j)];
                        ra_node2->label = labels[dimTran(i, j - 1)];
                        if (ralist[labels[dimTran(i, j)]].insert(ra_node2)) {
                            free_ra_list = old_ra_freelist;
                            temp_count -= 2;
                        } else
                            ralist[labels[dimTran(i, j - 1)]].insert(ra_node1);
                    }
                }

            // 2.Treat each region Ri as a disjoint set
            for (int i = 0; i < region_count; i++) {
                RAList *neighbor = ralist[i].next;
                while (neighbor) {
                    if (colorDistance(&mode[3 * i], &mode[3 * neighbor->label]) < color_radius * color_radius) {
                        int i_can_el = i, neigh_can_el = neighbor->label;
                        while (ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                        while (ralist[neigh_can_el].label != neigh_can_el) neigh_can_el = ralist[neigh_can_el].label;
                        if (i_can_el < neigh_can_el)
                            ralist[neigh_can_el].label = i_can_el;
                        else {
                            //ralist[ralist[i_can_el].label].label = i_can_el;
                            ralist[i_can_el].label = neigh_can_el;
                        }
                    }
                    neighbor = neighbor->next;
                }
            }
            // 3. Union Find
            for (int i = 0; i < region_count; i++) {
                int i_can_el = i;
                while (ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                ralist[i].label = i_can_el;
            }
            // 4. Traverse joint sets, relabeling image.
            int *mode_point_count_buffer = new int[region_count];
            memset(mode_point_count_buffer, 0, region_count * sizeof(int));
            float *mode_buffer = new float[region_count * 3];
            int *label_buffer = new int[region_count];

            for (int i = 0; i < region_count; i++) {
                label_buffer[i] = -1;
                mode_buffer[i * 3 + 0] = 0;
                mode_buffer[i * 3 + 1] = 0;
                mode_buffer[i * 3 + 2] = 0;
            }
            for (int i = 0; i < region_count; i++) {
                int i_can_el = ralist[i].label;
                mode_point_count_buffer[i_can_el] += mode_point_count[i];
                for (int k = 0; k < 3; k++)
                    mode_buffer[i_can_el * 3 + k] += mode[i * 3 + k] * mode_point_count[i];
            }
            int label = -1;
            for (int i = 0; i < region_count; i++) {
                int i_can_el = ralist[i].label;
                if (label_buffer[i_can_el] < 0) {
                    label_buffer[i_can_el] = ++label;

                    for (int k = 0; k < 3; k++)
                        mode[label * 3 + k] = (mode_buffer[i_can_el * 3 + k]) / (mode_point_count_buffer[i_can_el]);

                    mode_point_count[label] = mode_point_count_buffer[i_can_el];
                }
            }
            region_count = label + 1;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    labels[dimTran(i, j)] = label_buffer[ralist[labels[dimTran(i, j)]].label];

            delete[] mode_buffer;
            delete[] mode_point_count_buffer;
            delete[] label_buffer;

            //Destroy RAM
            delete[] ralist;
            delete[] raPool;

            delta_region_count = old_region_count - region_count;
            old_region_count = region_count;
            cout << "Mean Shift(TransitiveClosure):" << region_count << endl;
        }

    }
    /**
     * @brief 多线程实现union_find
     * @return labels会更新
     */
    void PthreadUnionFind::unionFindMultiThread() {

        for (int counter = 0, delta_region_count = 1; counter < 5 && delta_region_count > 0; counter++) {
            pthread_t *thread_pool = new pthread_t[THREAD_NUM];
            PthreadData *pthread_data_array = new PthreadData[THREAD_NUM];
            RAList *ralist = new RAList[region_count];
            RAList *ralist_array = new RAList[THREAD_NUM * region_count];
            int block_size = (width * height) / THREAD_NUM;
            void *thread_status;
            int thread_res;
            int old_region_count = region_count;
            for (int i = 0; i < region_count; i++) {
                ralist[i].label = i;
                ralist[i].next = NULL;
            }
            for (int i = 0; i < THREAD_NUM * region_count; i++) {
                ralist_array[i].label = i % region_count;
                ralist_array[i].next = NULL;
            }
            for (int i = 0; i < THREAD_NUM; i++) {
                int end = (i + 1) * block_size > width * height ? height * width - 1 : (i + 1) * block_size - 1;
                pthread_data_array[i] = new PthreadData(i * block_size, end, width, height, labels,
                                                        &ralist_array[i * region_count]);
            }

            for (int i = 0; i < THREAD_NUM; i++) {
                thread_res = pthread_create(&thread_pool[i], NULL, threadInsert, &pthread_data_array[i]);
                if (thread_res != 0) {
                    perror("Thread creation failed");
                    exit(EXIT_FAILURE);
                }
            }

            for (int i = 0; i < THREAD_NUM; i++) {
                thread_res = pthread_join(thread_pool[i], &thread_status);
                if (thread_res != 0) {
                    perror("Thread creation failed");
                    exit(EXIT_FAILURE);
                }
            }

            ///boundary analysis
            for (int i = 0; i < THREAD_NUM - 1; i++) {
                int end = (i + 1) * block_size > width * height ? height * width - 1 : (i + 1) * block_size - 1;
                for (int j = 0; j < width; j++) {
                    if (colorDistance(&mode[3 * (end - j)], &mode[3 * (end - j + width)]) <
                        color_radius * color_radius) {
                        RAList *ra_node1 = new RAList();
                        RAList *ra_node2 = new RAList();
                        ra_node1->label = labels[end - j];
                        ra_node2->label = labels[end - j + width];
                        if (ralist_array[i * region_count + labels[end - j]].insert(ra_node2) != 0) {
                            ralist_array[i * region_count + labels[end - j + width]].insert(ra_node1);
                        }
                    }
                }
            }


            for (int i = 0; i < THREAD_NUM * region_count; i++) {
                ralist[i % region_count] = *(ralist[i % region_count].merge(&ralist[i % region_count],
                                                                            &ralist_array[i]));
            }



            /// 2.Treat each region Ri as a disjoint set
            for (int i = 0; i < region_count; i++) {
                RAList *neighbor = ralist[i].next;
                while (neighbor) {
                    if (colorDistance(&mode[3 * i], &mode[3 * neighbor->label]) < color_radius * color_radius) {
                        int i_can_el = i, neigh_can_el = neighbor->label;
                        while (ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                        while (ralist[neigh_can_el].label != neigh_can_el) neigh_can_el = ralist[neigh_can_el].label;
                        if (i_can_el < neigh_can_el)
                            ralist[neigh_can_el].label = i_can_el;
                        else {
                            ///ralist[ralist[i_can_el].label].label = i_can_el;
                            ralist[i_can_el].label = neigh_can_el;
                        }
                    }
                    neighbor = neighbor->next;
                }
            }
            /// 3. Union Find
            for (int i = 0; i < region_count; i++) {
                int i_can_el = i;
                while (ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                ralist[i].label = i_can_el;
            }
            /// 4. Traverse joint sets, relabeling image.
            int *mode_point_count_buffer = new int[region_count];
            memset(mode_point_count_buffer, 0, region_count * sizeof(int));
            float *mode_buffer = new float[region_count * 3];
            int *label_buffer = new int[region_count];

            for (int i = 0; i < region_count; i++) {
                label_buffer[i] = -1;
                mode_buffer[i * 3 + 0] = 0;
                mode_buffer[i * 3 + 1] = 0;
                mode_buffer[i * 3 + 2] = 0;
            }
            for (int i = 0; i < region_count; i++) {
                int i_can_el = ralist[i].label;
                mode_point_count_buffer[i_can_el] += mode_point_count[i];
                for (int k = 0; k < 3; k++)
                    mode_buffer[i_can_el * 3 + k] += mode[i * 3 + k] * mode_point_count[i];
            }
            int label = -1;
            for (int i = 0; i < region_count; i++) {
                int i_can_el = ralist[i].label;
                if (label_buffer[i_can_el] < 0) {
                    label_buffer[i_can_el] = ++label;

                    for (int k = 0; k < 3; k++)
                        mode[label * 3 + k] = (mode_buffer[i_can_el * 3 + k]) / (mode_point_count_buffer[i_can_el]);

                    mode_point_count[label] = mode_point_count_buffer[i_can_el];
                }
            }
            region_count = label + 1;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    labels[dimTran(i, j)] = label_buffer[ralist[labels[dimTran(i, j)]].label];

            delete[] mode_buffer;
            delete[] mode_point_count_buffer;
            delete[] label_buffer;

            //Destroy RAM
            delete[] ralist;
            delete[] thread_pool;
            delete[] pthread_data_array;
            delete[] ralist_array;

            delta_region_count = old_region_count - region_count;
            old_region_count = region_count;
            cout << "Mean Shift(TransitiveClosure):" << region_count << endl;
        }
    }

    /**
     * @brief 二维到一维的转换
     *
     * @param i，j
     * @return int
     * @retval 一维的对应index
     */
    int PthreadUnionFind::dimTran(int i, int j) {
        return i * width + j;
    }

    /**
     * @brief 计算像素距离
     *
     * @param const float *a
     * @param const float *b
     * @return distance
     */
    float PthreadUnionFind::colorDistance(const float *a, const float *b) {
        float l = a[0] - b[0], u = a[1] - b[1], v = a[2] - b[2];
        return l * l + u * u + v * v;
    }

    /**
     * @brief 获取labels
     * @return labels
     */
    int *PthreadUnionFind::getResultLabels() {
        return labels;
    }
}

using namespace CuMeanShift;
void *threadInsert(void *arg) {
    PthreadData* data = (PthreadData*)arg;

    for (int i =data->start; i <= data->end; i++) {
        if (i >= data->width&&i<data->width*data->height&&data->labels[i] != data->labels[i - data->width]) {
            RAList* ra_node1 = new RAList();
            RAList* ra_node2 = new RAList();
            ra_node1->label = data->labels[i];
            ra_node2->label = data->labels[i - data->width];
            int insert_res=(data->ralist)[ra_node1->label].insert(ra_node2);

            if (insert_res != 0) {

                (data->ralist)[ra_node2->label].insert(ra_node1);

            }
        }
        if (i%data->width!=0&&i<data->width*data->height&&data->labels[i] != data->labels[i - 1]) {
            RAList* ra_node1 = new RAList();
            RAList* ra_node2 = new RAList();
            ra_node1->label = data->labels[i];
            ra_node2->label = data->labels[i - 1];
            int insert_res=(data->ralist)[ra_node1->label].insert(ra_node2);
            if (insert_res != 0) {
                (data->ralist)[ra_node2->label].insert(ra_node1);
            }
        }
    }
    pthread_exit(0);
    return NULL;
}


