#pragma once
#include <net.h>
namespace seedvr2::engine {
int register_fp32_softmax(ncnn::Net &net);
// Dense FP32 attention with pre-scaled Q/K, no mask, cache, or grouped heads.
ncnn::Layer *create_fp32_sdpa();
}
