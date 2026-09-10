#pragma once
#include <net.h>
namespace seedvr2::engine {
// Scoped to FP32 DiT token projections. CPU keeps the upstream implementation.
int register_dit_linear(ncnn::Net &net);
}
