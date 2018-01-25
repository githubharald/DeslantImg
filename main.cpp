#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "DeslantImg.hpp"


int main()
{
	// read grayscale image
	const cv::Mat img = cv::imread("data/test.png", cv::IMREAD_GRAYSCALE);
	
	// deslant it
	cv::Mat res = deslantImg(img, 255);

	// and save the result
	cv::imwrite("out.png", res);

	return 0;
}
