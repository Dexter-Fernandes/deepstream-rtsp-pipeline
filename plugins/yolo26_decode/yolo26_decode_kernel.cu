#include "yolo26_decode_kernel.cuh"

// One thread per detection.  Converts xyxy pixel coords → xywh and copies
// confidence + class_id unchanged.  Input and output are both [N, 6] float32
// laid out row-major in device memory.
__global__ void yolo26_decode_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_dets)
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_dets) return;

    const float* row_in  = in  + i * 6;
    float*       row_out = out + i * 6;

    const float x1   = row_in[0];
    const float y1   = row_in[1];
    const float x2   = row_in[2];
    const float y2   = row_in[3];
    const float conf = row_in[4];
    const float cls  = row_in[5];

    row_out[0] = x1;        // left
    row_out[1] = y1;        // top
    row_out[2] = x2 - x1;  // width
    row_out[3] = y2 - y1;  // height
    row_out[4] = conf;
    row_out[5] = cls;
}

void yolo26_decode(
    const float* d_in,
    float*       d_out,
    int          n_dets,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (n_dets + threads - 1) / threads;
    yolo26_decode_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n_dets);
}
