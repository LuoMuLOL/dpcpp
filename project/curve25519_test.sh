#!/bin/bash

# $1 传入的第一个参数,即使用./run.sh simple 或 ./run.sh cuda调用不同代码
if [ "$1" == "simple" ]
then
    echo "普通版本"
    export PATH=$DPCPP_CPDIR/build/bin:$PATH
    export LD_LIBRARY_PATH=$DPCPP_CPDIR/build/lib:$LD_LIBRARY_PATH
    ./build/build/Linux-DPCplusplus-clang/deps/curve25519/curve25519
elif [ "$1" == "cuda" ]
then
    echo "cuda版本"
    export PATH=$DPCPP_CUDA_CPDIR/build/bin:$PATH
    export LD_LIBRARY_PATH=$DPCPP_CUDA_CPDIR/build/lib:$LD_LIBRARY_PATH
    ./build/build/Linux-DPCplusplus-clang-cuda/deps/curve25519/curve25519
    exit 1
fi
exit 0