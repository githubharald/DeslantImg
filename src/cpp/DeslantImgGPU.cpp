#include "DeslantImgGPU.hpp"
#include <opencv2/imgproc.hpp>


namespace htr
{

	cv::Mat deslantImg(const cv::Mat& img, const int bgcolor, CLWrapper& clWrapper)
	{
		// must be grayscale img
		assert(img.channels() == 1);

		// map to binary image
		cv::Mat imgBW;
		cv::resize(img, imgBW, cv::Size(static_cast<int>(clWrapper.imgW), static_cast<int>(clWrapper.imgH)), 0.0, 0.0, cv::INTER_NEAREST);
		cv::threshold(imgBW, imgBW, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

		// use float datatype (single channel, 32bit)
		imgBW.convertTo(imgBW, CV_32FC1, 1.0 / 255.0);

		// upload image data to GPU
		clWrapper.setData(imgBW);

		// compute best shearing value on GPU
		const float alpha = clWrapper.compute();

		// setting transformation matrix
		const float shiftX = std::max(-alpha*img.size().height, 0.0f);
		cv::Mat transform;
		transform = cv::Mat(2, 3, CV_32F);
		transform.at<float>(0, 0) = 1;
		transform.at<float>(0, 1) = alpha;
		transform.at<float>(0, 2) = shiftX;
		transform.at<float>(1, 0) = 0;
		transform.at<float>(1, 1) = 1;
		transform.at<float>(1, 2) = 0;
		cv::Size size = cv::Size(img.size().width + static_cast<int>(std::ceil(std::abs(alpha*img.size().height))), img.size().height);;

		// shear image and return
		cv::Mat imgSheared;
		cv::warpAffine(img, imgSheared, transform, size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(255));
		return imgSheared;
	}

}

