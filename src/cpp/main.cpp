#ifdef USE_GPU
#include "DeslantImgGPU.hpp"
#else
#include "DeslantImgCPU.hpp"
#endif
#include <opencv2/imgcodecs.hpp>
#include <iostream>


int main() 
{
	// load input image
	const cv::Mat img1 = cv::imread("data/test1.png", cv::IMREAD_GRAYSCALE);
	const cv::Mat img2 = cv::imread("data/test2.png", cv::IMREAD_GRAYSCALE);
	
	
#ifdef USE_GPU 
	// deslant on GPU
	htr::CLWrapper clWrapper; // setup OpenCL, the same instance should be used for all following calls to deslantImg
	const cv::Mat res1 = htr::deslantImg(img1, 255, clWrapper);
	const cv::Mat res2 = htr::deslantImg(img2, 255, clWrapper);
#else
	// deslant on CPU
	cv::Mat res1 = htr::deslantImg(img1, 255);
	cv::Mat res2 = htr::deslantImg(img2, 255);
#endif

	// write results to file
	cv::imwrite("out1.png", res1);
	cv::imwrite("out2.png", res2);

	return 0;
}

