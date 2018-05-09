#pragma once


// uncomment next line to profile kernel1 and kernel2 and write result to variable timeKernel1/2
#define USE_GPU_PROFILING


#include <string>
#include <opencv2/core.hpp>
#include <CL/cl.h>


namespace htr
{

	class CLWrapper
	{
	public:
		// CTOR
		CLWrapper();

		// DTOR
		~CLWrapper();

		// set input image
		void setData(const cv::Mat& img);

		// compute shearing value to deslant image
		float compute();

		// some constants regarding data
		const size_t maxShearedW = 1024;
		const size_t imgH = 128;
		const size_t imgW = maxShearedW - imgH;
		const size_t numAlphaValues = 9;
		const size_t numWorkgroups = 4; 
		const size_t sizeWorkgroup = maxShearedW / numWorkgroups;

#ifdef USE_GPU_PROFILING
		// profiling info
		cl_ulong timeKernel1 = 0, timeKernel2 = 0;
#endif

	private:
		CLWrapper(CLWrapper&) = delete;
		CLWrapper& operator=(CLWrapper&) = delete;

		void setupDevice();
		void setupKernel();
		std::string buildErrorString();
		void releaseData();

		// plattform, device, context and queue
		cl_platform_id platform = nullptr;
		cl_device_id device = nullptr;
		cl_context context = nullptr;
		cl_command_queue queue = nullptr;
		// program and kernel
		cl_program program = nullptr;
		cl_kernel kernel1 = nullptr;
		cl_kernel kernel2 = nullptr;
		
		// data for pass 1
		cl_mem dataIn1 = nullptr;
		cl_mem dataOut1 = nullptr;
		// data for pass 2
		cl_mem dataIn2 = nullptr;
		cl_mem dataOut2 = nullptr;
		// remember if buffers already allocated
		bool dataAlloc = false;
	};

}

