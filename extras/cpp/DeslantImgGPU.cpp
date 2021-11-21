#include "DeslantImgGPU.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


namespace htr
{
	// resize to fit max size
	cv::Mat resizeImg(const cv::Mat& img, const int maxW, const int maxH)
	{
		cv::Mat res;

		// img size
		const int w = img.cols, h = img.rows;

		// resizing needed?
		if (w > maxW || h > maxH)
		{
			const double fx = w / double(maxW), fy = h / double(maxH);
			const double f = std::max(fx, fy);
			const int wn = int(w / f), hn = int(h / f);
			cv::resize(img, res, cv::Size(wn, hn));
		}
		else
		{
			res = img;
		}

		return res;
	}


	cv::Mat deslantImg(const cv::Mat& img, const int bgcolor, CLWrapper& clWrapper)
	{
		// must be grayscale img
		assert(img.channels() == 1);

		// resize to fit 
		const cv::Mat imgResized = resizeImg(img, static_cast<int>(clWrapper.imgW), static_cast<int>(clWrapper.imgH));

		// map to binary image
		cv::Mat imgBW;
		cv::threshold(imgResized, imgBW, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
		
		// copy into target image and convert to float image (32 bit, 1 channel)
		cv::Mat targetImg=cv::Mat::zeros(cv::Size(static_cast<int>(clWrapper.imgW), static_cast<int>(clWrapper.imgH)), CV_8UC1);
		imgBW.copyTo(targetImg(cv::Rect(0, 0, imgBW.size().width, imgBW.size().height)));

		// upload image data to GPU
		clWrapper.setData(targetImg);

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
		cv::Mat imgDeslanted;
		cv::warpAffine(img, imgDeslanted, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(bgcolor));

		return imgDeslanted;
	}

}

