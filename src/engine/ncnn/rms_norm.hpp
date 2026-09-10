#pragma once
#include <net.h>
namespace seedvr2::engine {
// FP32-B, non-affine 3B DiT RMSNorm. CPU graphs retain the ncnn CPU layer.
int register_dit_rms_norm(ncnn::Net &net);
}
