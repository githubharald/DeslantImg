#include "DeslantImgCPU.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <assert.h>
#include <algorithm>


namespace htr
{

	// data structure for internal usage
	struct Result
	{
		float sum_alpha = 0.0f;
		cv::Mat transform;
		cv::Size size;
		bool operator < (const Result& rhs) const { return sum_alpha < rhs.sum_alpha; }
	};


	// deslant image containing text (remove italic style)
	// * img: grayscale image containing text
	// * bgcolor: empty space in result image is filled with this color (0...255)
	// * returns: deslanted image
	cv::Mat deslantImg(const cv::Mat& img, const int bgcolor)
	{
		// must be grayscale img
		assert(img.channels() == 1);

		// calc binary img
		cv::Mat imgBW;
		cv::threshold(img, imgBW, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

		// list of alpha values controlling shear transform (search space)
		std::vector<float> alphaVals = { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 };

		// variable names according to paper Vinciarelli
		std::vector<int> sum_alpha(alphaVals.size(), 0);

		// go through alpha values to see which is best for deslanting
		std::vector<Result> results;
		for (size_t i = 0; i < alphaVals.size(); ++i)
		{
			Result result;

			const float alpha = alphaVals[i];
			const float shiftX = std::max(-alpha*imgBW.rows, 0.0f);
			result.size = cv::Size(imgBW.cols + static_cast<int>(std::ceil(std::abs(alpha*imgBW.rows))), imgBW.rows);

			// transformation matrix (2x3)
			// x'=M00*x+M01*y+M02
			// y'=M10*x+M11*y+M12
			result.transform = cv::Mat(2, 3, CV_32F);
			result.transform.at<float>(0, 0) = 1;
			result.transform.at<float>(0, 1) = alpha;
			result.transform.at<float>(0, 2) = shiftX;
			result.transform.at<float>(1, 0) = 0;
			result.transform.at<float>(1, 1) = 1;
			result.transform.at<float>(1, 2) = 0;
			cv::Mat imgSheared;

			// shear image
			cv::warpAffine(imgBW, imgSheared, result.transform, result.size, cv::INTER_NEAREST);

			// go through all cols
			for (int x = 0; x < imgSheared.cols; ++x)
			{
				// calc h_alpha and delta_y_alpha 
				std::vector<int> fgIndices;
				for (int y = 0; y < imgSheared.rows; ++y)
				{
					if (imgSheared.at<unsigned char>(y, x))
					{
						fgIndices.push_back(y);
					}
				}

				// no fg pixels ... nothing to do
				if (fgIndices.empty())
				{
					continue;
				}

				// variable names according to paper Vinciarelli
				int h_alpha = static_cast<int>(fgIndices.size()); // number of fg pixels in col
				int delta_y_alpha = fgIndices.back() - fgIndices.front() + 1; // distance between first and last fg pixel in col (+1 to avoid getting Float.Inf)

				// if H_alpha (=h_alpha/delta_y_alpha) == 1
				if (h_alpha == delta_y_alpha)
				{
					result.sum_alpha += h_alpha*h_alpha;
				}
			}

			results.push_back(result);
		}

		// use best result to transform image
		Result bestResult = *std::max_element(results.begin(), results.end());
		cv::Mat imgDeslanted;
		cv::warpAffine(img, imgDeslanted, bestResult.transform, bestResult.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(bgcolor));

		return imgDeslanted;
	}

}

