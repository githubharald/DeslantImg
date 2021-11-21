#pragma once
#include <opencv2/core.hpp>


namespace htr
{

	// deslant image containing text (remove italic style)
	// * img: grayscale image containing text
	// * bgcolor: empty space in result image is filled with this color (0...255)
	// * returns: deslanted image
	cv::Mat deslantImg(const cv::Mat& img, const int bgcolor, const float lower_bound=-1.0, const float upper_bound=1.0);

}

