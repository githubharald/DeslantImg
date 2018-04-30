#pragma once


// uncomment next line to profile kernel1 and kernel2 and write result to variable timeKernel1/2
//#define USE_GPU_PROFILING


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
		const size_t imgW = 1024, imgH = 128; // use fixed input size to avoid buffer reallocation
		const size_t numAlphaValues = 9;

#ifdef USE_GPU_PROFILING
		// profiling info
		cl_ulong timeKernel1=0, timeKernel2=0;
#endif

	private:
		CLWrapper(CLWrapper&) = delete;
		CLWrapper& operator=(CLWrapper&) = delete;

		void setupDevice();
		void setupKernel(int programID);
		std::string buildErrorString(int programID);
		void releaseData();

		// plattform, device, context and queue
		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		cl_command_queue queue;
		// kernel for pass 1
		cl_program program1;
		cl_kernel kernel1;
		// kernel for pass 2
		cl_program program2;
		cl_kernel kernel2;
		
		// data for pass 1
		cl_mem dataIn1 = nullptr;
		cl_mem dataOut1 = nullptr;
		// data for pass 2
		cl_mem dataIn2 = nullptr;
		cl_mem dataOut2 = nullptr;
		// remember if buffers already allocated
		bool dataAlloc = false;
		// max width of a sheared img
		size_t maxShearedW = 0;
		// result buffer
		std::vector<float> resBuffer = std::vector<float>(numAlphaValues, 0.0f);
	};

}

