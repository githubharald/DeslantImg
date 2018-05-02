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


const size_t NUM_ALPHA_VALS = 9;
float alphaVals[NUM_ALPHA_VALS] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };


namespace htr
{

	CLWrapper::CLWrapper()
	{
		maxShearedW = static_cast<size_t>(imgW - alphaVals[0] * imgH);
		assert(maxShearedW % 16 == 0); // must be multiple of 16 such that int16 can be used in kernel
		setupDevice();
		setupKernel(1);
		setupKernel(2);
	}


	CLWrapper::~CLWrapper()
	{
		// data
		releaseData();

		// kernel
		clReleaseKernel(kernel1);
		clReleaseKernel(kernel2);

		// program
		clReleaseProgram(program1);
		clReleaseProgram(program2);

		// queue, context
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


	std::string CLWrapper::buildErrorString(int programID)
	{
		// create char buf
		size_t logSize = 0;
		clGetProgramBuildInfo(programID == 1 ? program1 : program2, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		std::vector<char> charBuf(logSize, 0);

		// read into char buf and return as string
		clGetProgramBuildInfo(programID == 1 ? program1 : program2, device, CL_PROGRAM_BUILD_LOG, logSize, charBuf.data(), NULL);
		return std::string(charBuf.begin(), charBuf.end());
	}


	void CLWrapper::setupKernel(int programID)
	{
		// read kernel
		std::ifstream f(programID == 1 ? "src/cl/ProcessColumns.cl" : "src/cl/SumColumns.cl");
		const std::string strKernel((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		const size_t kernelSize = strKernel.size();
		const char* ptrToStrKernel = strKernel.data();

		// create program
		int err = 0;
		cl_program program = clCreateProgramWithSource(context, 1, &ptrToStrKernel, &kernelSize, &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the program");
		}
		if (programID == 1)
		{
			program1 = program;
		}
		else
		{
			program2 = program;
		}

		// build program
		err = clBuildProgram(programID == 1 ? program1 : program2, 0, NULL, NULL, NULL, NULL);
		if (err < 0)
		{
			throw std::runtime_error(buildErrorString(programID).c_str());
		}

		// kernel
		cl_kernel kernel = clCreateKernel(programID == 1 ? program1 : program2, programID == 1 ? "processColumns" : "sumColumns", &err);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't create the kernel");
		}
		if (programID == 1)
		{
			kernel1 = kernel;
		}
		else
		{
			kernel2 = kernel;
		}
	}


	void CLWrapper::releaseData()
	{
		if (!dataIn1)
		{
			clReleaseMemObject(dataIn1);
		}

		if (!dataOut1) // dataOut1 == dataIn2
		{
			clReleaseMemObject(dataOut1);
		}

		if (!dataOut2)
		{
			clReleaseMemObject(dataOut1);
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

			// in 1
			cl_image_format imgFormat;
			imgFormat.image_channel_data_type = CL_FLOAT;
			imgFormat.image_channel_order = CL_LUMINANCE;
			dataIn1 = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFormat, imgW, imgH, 0, img.data, &err);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't create a buffer object");
			}

			// out 1: we need enough columns to hold result for sheared img with max. shear value
			//dataOut1 = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &imgFormat, maxShearedW, numAlphaValues, 0, NULL, &err);
			dataOut1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * maxShearedW * numAlphaValues, NULL, &err);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't create a buffer object");
			}

			// in 2
			dataIn2 = dataOut1;

			// out 2
			dataOut2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*numAlphaValues, NULL, &err);
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
			const cl_int maxShearedWAsInt = static_cast<cl_int>(maxShearedW);
			err = clSetKernelArg(kernel1, 1, sizeof(cl_int), &maxShearedWAsInt);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}

			// arg2
			err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &dataOut1);
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
			err = clSetKernelArg(kernel2, 1, sizeof(cl_int), &maxShearedWAsInt);
			if (err < 0)
			{
				throw std::runtime_error("Couldn't set the kernel argument");
			}

			// arg2
			err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), &dataOut2);
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
		const size_t workItems1[2] = { maxShearedW, numAlphaValues };
		int err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, workItems1, NULL, 0, NULL, &eventKernel1);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't enqueue the kernel execution command");
		}

		// execute kernel2
		cl_event eventKernel2;
		const size_t workItems2[1] = { numAlphaValues };
		err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, workItems2, NULL, 0, NULL, &eventKernel2);
		if (err < 0)
		{
			throw std::runtime_error("Couldn't enqueue the kernel execution command");
		}

		// read result
		err = clEnqueueReadBuffer(queue, dataOut2, CL_TRUE, 0, sizeof(cl_int)*numAlphaValues, resBuffer.data(), 0, NULL, NULL);
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

		size_t maxAlphaIdx = 0;
		cl_int maxAlphaVal = 0;
		for (size_t i = 0; i < numAlphaValues; ++i)
		{
			if (resBuffer[i] > maxAlphaVal)
			{
				maxAlphaVal = resBuffer[i];
				maxAlphaIdx = i;
			}
		}

		// put into cv::Mat
		return -alphaVals[maxAlphaIdx];
	}

}

