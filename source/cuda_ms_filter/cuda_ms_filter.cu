#include <cuda_ms_filter/cuda_ms_filter.h>
__device__ bool islessequal(float a ,float b) {
    return a < b;
}
__global__ void compute_mean_shift(
        float4* img,
        float4* dst,
        int width,
        int height,
        int dis_range,
        float color_range,
        float min_shift,
        int max_iter)
{
    const int pad_width = width + dis_range * 2;
    const int pad_height = height + dis_range * 2;
    int2 global_id = make_int2(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);
    int2 relative_id = global_id + make_int2(dis_range, dis_range);

    if (global_id.x >= width || global_id.y >= height)
        return;

    float2 id = make_float2(0, 0);
    float4 color = make_float4(0, 0, 0, 0);
    float  mask;
    float  sum_num = 0;

    float4 cur_color, pre_color;
    float2 cur_id, pre_id;
    int from_idx, to_idx, from_idy, to_idy;

    cur_id = make_float2(relative_id.x, relative_id.y);
    cur_color = img[relative_id.x + relative_id.y * pad_width];
    pre_id = cur_id;
    pre_color = cur_color;

    //previous and current center point
    from_idx = cur_id.x - dis_range;
    to_idx = cur_id.x + dis_range + 1;
    from_idy = cur_id.y - dis_range;
    to_idy = cur_id.y + dis_range + 1;

    for (int iters = 0; iters < max_iter; iters++) {
        for (int tmp_idx = from_idx; tmp_idx < to_idx; tmp_idx++) {
            for (int tmp_idy = from_idy; tmp_idy < to_idy; tmp_idy++) {

                float4 tmp_color;
                tmp_color = img[tmp_idx + tmp_idy * pad_width];
                float4 tmp = tmp_color - cur_color;
                float d_color = dot(tmp, tmp);

                /// you can consider mask as if
                mask = (islessequal(d_color, color_range));
                color += tmp_color * mask;
                id += make_float2(tmp_idx, tmp_idy) * mask;
                sum_num += mask;

            }
        }

        /// calculate average
        bool flag = sum_num == 0;
        if (flag) break;
        cur_id = id / sum_num;
        cur_color = color / sum_num;

        //calculate shift
        float2 d_id = cur_id - pre_id;
        float4 d_color = cur_color - pre_color;
        float shift = dot(d_id, d_id) + dot(d_color, d_color);
        flag = islessequal(shift, min_shift);
        if (flag) break;

        from_idx = cur_id.x - dis_range;
        to_idx = cur_id.x + dis_range + 1;
        from_idy = cur_id.y - dis_range;
        to_idy = cur_id.y + dis_range + 1;
        pre_id = cur_id;
    }
    dst[global_id.x + global_id.y * width] = cur_color;
}