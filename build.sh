#!/bin/bash

if [ "$1" == "gpu" ]; then
	echo "Compiling for GPU using OpenCL. OpenCL must be installed."
	LIBCL="-lOpenCL"
	DEFGPU="-D USE_GPU"
	WRAPPERFILE="src/cpp/CLWrapper.cpp"
	IMPLFILE="src/cpp/DeslantImgGPU.cpp"
	
else
	echo "Compiling for CPU."
	IMPLFILE="src/cpp/DeslantImgCPU.cpp"
fi

g++ --std=c++11 $DEFGPU src/cpp/main.cpp $IMPLFILE $WRAPPERFILE `pkg-config --cflags --libs opencv` $LIBCL -o DeslantImg

