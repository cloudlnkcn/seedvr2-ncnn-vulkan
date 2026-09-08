#!/bin/sh
set -eu
test ! -e /home/mingshi/Project/AI/pnnx_torchdynamo/seedvr2-ncnn-vulkan/src/engine/ncnn/image.cpp
test "$(wc -l < /proc/net/route)" -eq 1
readlink /proc/self/ns/net > /tmp/work/network-namespace.txt
cat /proc/net/route > /tmp/work/network-routes.txt
'/opt/SeedVR2 空间'/bin/seedvr2 engine self-test --backend cpu > /tmp/work/cpu.json
cmake -S /tmp/work/consumer -B /tmp/work/build -G Ninja -DCMAKE_PREFIX_PATH='/opt/SeedVR2 空间' > /tmp/work/sdk-build.log 2>&1
cmake --build /tmp/work/build >> /tmp/work/sdk-build.log 2>&1
/tmp/work/build/seedvr2-sdk-example > /tmp/work/sdk-smoke.json
/tmp/work/build/seedvr2-sdk-example image '/opt/SeedVR2 空间/models/image' '/tmp/work/输入 照片.jpg' '/tmp/work/结果 自然图像' vulkan 256 > /tmp/work/image.json
