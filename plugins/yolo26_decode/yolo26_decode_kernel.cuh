#pragma once
#include <cuda_runtime.h>

// Convert a batch of YOLO26n detections from xyxy → xywh in device memory.
// Both buffers are [n_dets, 6] float32 (row-major).
void yolo26_decode(
    const float* d_in,
    float*       d_out,
    int          n_dets,
    cudaStream_t stream);
