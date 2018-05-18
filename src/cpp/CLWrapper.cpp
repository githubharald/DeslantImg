#include "CLWrapper.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <cassert>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


namespace htr
{

	CLWrapper::CLWrapper()
	{
		setupDevice();
		setupKernel();
	}


	CLWrapper::~CLWrapper()
	{
		// data
		releaseData();

		// kernel
		clReleaseKernel(kernel1);
		clReleaseKernel(kernel2);

		// program
		clReleaseProgram(program);

		// queue, context, device
		clReleaseCommandQueue(queue);
		clReleaseContext(context);

	}


	void CLWrapper::setupDevice()
	{
		// platform
		int err = clGetPlatformIDs(1, &platform, NULL);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't find any platforms");
		}

		// device
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't find any devices");
		}

		// context
		context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create a context");
		}

		// queue
#ifdef USE_GPU_PROFILING
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#else
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the command queue");
		}
	}


	std::string CLWrapper::buildErrorString()
	{
		// create char buf
		size_t logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		std::vector<char> charBuf(logSize, 0);

		// read into char buf and return as string
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, charBuf.data(), NULL);
		return std::string(charBuf.begin(), charBuf.end());
	}


	void CLWrapper::setupKernel()
	{
		// read kernel
		std::ifstream f("src/cl/Kernel.cl");
		const std::string strKernel((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		const size_t kernelSize = strKernel.size();
		const char* ptrToStrKernel = strKernel.data();

		// create program
		int err = 0;
		program = clCreateProgramWithSource(context, 1, &ptrToStrKernel, &kernelSize, &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the program");
		}		

		// build program
		err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (err < 0)
		{
			throw std::runtime_error(buildErrorString().c_str());
		}

		// kernel
		kernel1 = clCreateKernel(program,"processColumns", &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the kernel");
		}

		// kernel2
		kernel2 = clCreateKernel(program, "sumColumns", &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the kernel");
		}
		
	}


	void CLWrapper::releaseData()
	{
		if (dataAlloc)
		{
			clReleaseMemObject(dataIn1);
			clReleaseMemObject(dataOut1);
			clReleaseMemObject(dataOut2);
		}
	}


	void CLWrapper::setData(const cv::Mat& img)
	{
		// resize to fixed size
		assert(imgW == img.size().width && imgH == img.size().height);

		// allocate buffers only once
		int err = 0;
		if (!dataAlloc)
		{
			dataAlloc = true;

			// in 1: image with fixed size
			cl_image_format imgFormat;
			imgFormat.image_channel_data_type = CL_UNORM_INT8;
			imgFormat.image_channel_order = CL_LUMINANCE;
			dataIn1 = clCreateImage2D(context, CL_MEM_READ_ONLY, &imgFormat, imgW, imgH, 0, NULL, &err);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't create a buffer object");
			}

			// out 1: sum for each workgroup and shear angle
			dataOut1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * numWorkgroups * numAlphaValues, NULL, &err);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't create a buffer object");
			}

			// in 2: output of first kernel
			dataIn2 = dataOut1;

			// out 2: the shear angle which minimizes the slant of the text
			dataOut2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL, &err);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't create a buffer object");
			}

			// args for kernel1
			// arg0
			err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &dataIn1);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}

			// arg1
			err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), &dataOut1);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}

			// args for kernel2
			// arg0
			err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &dataIn2);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}

			// arg1
			err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &dataOut2);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}
		}
		
		// upload data, wait until finished such that img can be destroyed at any time
		size_t region[3] = {imgW, imgH, 1};
		size_t origin[3] = {0, 0, 0};
		err = clEnqueueWriteImage(queue, dataIn1, CL_TRUE, origin, region, 0, 0, img.data, 0, 0, 0);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't write input image");
		}
		
	}


	float CLWrapper::compute()
	{
		// execute kernel1
		cl_event eventKernel1;
		const size_t globalSize1[2] = { maxShearedW, numAlphaValues };
		const size_t localSize1[2] = { sizeWorkgroup, 1 };
		int err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, globalSize1, localSize1, 0, NULL, &eventKernel1);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't enqueue the kernel execution command");
		}

		// execute kernel2
		cl_event eventKernel2;
		const size_t globalSize2[1] = { numAlphaValues };
		const size_t localSize2[1] = { numAlphaValues };
		err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, globalSize2, localSize2, 0, NULL, &eventKernel2);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't enqueue the kernel execution command");
		}

		// read result
		cl_float bestAlphaVal=0.0f;
		err = clEnqueueReadBuffer(queue, dataOut2, CL_TRUE, 0, sizeof(cl_float), &bestAlphaVal, 0, NULL, NULL);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't enqueue the read buffer command");
		}

#ifdef USE_GPU_PROFILING
		// profiling info
		cl_ulong ts, te;
		clGetEventProfilingInfo(eventKernel1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ts, NULL);
		clGetEventProfilingInfo(eventKernel1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &te, NULL);
		timeKernel1 = te - ts;
		clGetEventProfilingInfo(eventKernel2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ts, NULL);
		clGetEventProfilingInfo(eventKernel2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &te, NULL);
		timeKernel2 = te - ts;
#endif

		clReleaseEvent(eventKernel1);
		clReleaseEvent(eventKernel2);

		return bestAlphaVal;
	}

}

