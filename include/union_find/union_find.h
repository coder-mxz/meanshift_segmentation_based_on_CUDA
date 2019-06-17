/**
 * @file union_find.h
 * @brief 多线程实现union_find的并行
 * @author mxz
 * @date 2019-6-15
 */


#ifndef MEANSHIFT_SEGMENTATION_BASED_ON_CUDA_MASTER_UNION_FIND_H
#define MEANSHIFT_SEGMENTATION_BASED_ON_CUDA_MASTER_UNION_FIND_H

#endif //MEANSHIFT_SEGMENTATION_BASED_ON_CUDA_MASTER_UNION_FIND_H


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

using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::ofstream;
using namespace cimg_library;
using namespace std;

namespace CuMeanShift {
    struct Point {
        int x, y;

        Point(int x, int y) : x(x), y(y) {}
    };

    class RAList {
    private:
        int exists;
        RAList *cur;
    public:
        int label;
        RAList *next;

    public:
        RAList(void);

        ~RAList(void);

        int insert(RAList *entry);

        RAList *merge(RAList *head1, RAList *head2);


    };


    RAList::RAList(void) {
        label = -1;
        next = 0;    ///NULL
    }

    RAList::~RAList(void) {}

    int RAList::insert(RAList *entry) {
        if (!next) {
            next = entry;
            entry->next = 0;
            return 0;
        }
        if (next->label > entry->label) {
            entry->next = next;
            next = entry;
            return 0;
        }
        exists = 0;
        cur = next;
        while (cur) {
            if (entry->label == cur->label) {
                exists = 1;
                break;
            } else if ((!(cur->next)) || (cur->next->label > entry->label)) {
                entry->next = cur->next;
                cur->next = entry;
                break;
            }
            cur = cur->next;
        }
        return (int) (exists);
    }


    RAList *RAList::merge(RAList *head1, RAList *head2) {
        if (head1 == NULL)return head2;
        if (head2 == NULL)return head1;
        RAList *res, *p;
        if (head1->label < head2->label) {
            res = head1;
            head1 = head1->next;
        } else if (head1->label > head2->label) {
            res = head2;
            head2 = head2->next;
        } else {
            res = head1;
            head1 = head1->next;
            head2 = head2->next;
        }
        p = res;

        while (head1 != NULL && head2 != NULL) {
            if (head1->label < head2->label) {
                p->next = head1;
                head1 = head1->next;
            } else if (head1->label > head2->label) {
                p->next = head2;
                head2 = head2->next;
            } else {
                p->next = head2;
                head1 = head1->next;
                head2 = head2->next;
            }
            p = p->next;
        }
        if (head1 != NULL)p->next = head1;
        else if (head2 != NULL)p->next = head2;
        return res;
    }


    struct PthreadData {

        int start, end, width, height;
        int *labels;
        RAList *ralist;

        PthreadData(int i_start = 0, int i_end = 0, int i_width = 0, int i_height = 0, int *i_labels = NULL,
                    RAList *i_ralist = NULL) {
            start = i_start;
            end = i_end;
            width = i_width;
            height = i_height;
            labels = i_labels;
            ralist = i_ralist;
        }

        PthreadData(PthreadData *new_data) {
            start = new_data->start;
            end = new_data->end;
            width = new_data->width;
            height = new_data->height;
            labels = new_data->labels;
            ralist = new_data->ralist;
        }
    };


    class PthreadUnionFind {
    private:
        int width, height, region_count;
        int *labels;
        int *new_labels;
        int *mode_point_count;
        float *mode;

    public:
        PthreadUnionFind(int *labels,
                         float *mode,
                         int *new_labels,
                         int *mode_point_count,

                         int width,
                         int height);

        PthreadUnionFind(CImg<uint8_t> &img, CImg<int> &labels);

        void unionFindOneThread();

        void unionFindMultiThread();

        int *getResultLabels();

    private:
        int dimTran(int i, int j);

        void setRegionCount();

        float colorDistance(const float *a, const float *b);

    };
}
