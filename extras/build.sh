#!/bin/bash

if [ "$1" == "gpu" ]; then
	echo "Compiling for GPU using OpenCL. OpenCL must be installed."
	LIBCL="-lOpenCL"
	DEFGPU="-D USE_GPU"
	WRAPPERFILE="cpp/CLWrapper.cpp"
	IMPLFILE="cpp/DeslantImgGPU.cpp"
	
else
	echo "Compiling for CPU."
	IMPLFILE="cpp/DeslantImgCPU.cpp"
fi

g++ --std=c++11 $DEFGPU cpp/main.cpp $IMPLFILE $WRAPPERFILE `pkg-config --cflags --libs opencv` $LIBCL -o DeslantImg

