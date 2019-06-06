//
// Created by iffi on 19-6-5.
//

#ifndef CUDA_MEANSHIFT_SEG_UTILS_H
#define CUDA_MEANSHIFT_SEG_UTILS_H

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define FLOOR(a, b) ((a) / (b))
#define CEIL(a, b) ( ((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL(a, b) * (b))
#define UNUSED(a)   (void)(a)
#define TIME_BEGIN(timer_num) \
    struct timeval timer_ ##timer_num## _start;\
    gettimeofday(&timer_ ##timer_num ##_start, NULL);

#define TIME_END(timer_num) \
    struct timeval timer_ ##timer_num## _end;\
    gettimeofday(&timer_ ##timer_num## _end, NULL);\
    {\
        double msec = (double)(timer_ ##timer_num## _end.tv_usec - timer_ ##timer_num## _start.tv_usec) / 1000 +\
                      (double)(timer_ ##timer_num## _end.tv_sec - timer_ ##timer_num## _start.tv_sec) * 1000;\
        printf(">>> timer_" #timer_num ":    %.3lf ms\n", msec);\
    }


#endif //CUDA_MEANSHIFT_SEG_UTILS_H
