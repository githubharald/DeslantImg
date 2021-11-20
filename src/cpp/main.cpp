#ifdef USE_GPU
#include "DeslantImgGPU.hpp"
#else
#include "DeslantImgCPU.hpp"
#endif
#include <opencv2/imgcodecs.hpp>
#include <iostream>


int main(int argc, const char *argv[])
{
	const cv::String opts =
		"{help h usage ? |      | print this message   }"
		"{@imagein       |-     | path name to read the input image from (or stdin) }"
		"{@imageout      |-     | path name to write the output image to (or stdout) }"
		"{lower_bound    |-1.0  | lower bound of shear values }"
		"{upper_bound    |1.0   | upper bound of shear values }"
		"{bg_color       |255   | color to fill the gaps of the sheared image that is returned }"
		;
	cv::CommandLineParser parser(argc, argv, opts);
#ifdef USE_GPU
	parser.about("DeslantImg GPU");
#else
	parser.about("DeslantImg CPU");
#endif
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	cv::String inpath = parser.get<cv::String>("@imagein");
	cv::String outpath = parser.get<cv::String>("@imageout");
	float lower_bound = parser.get<float>("lower_bound");
	float upper_bound = parser.get<float>("upper_bound");
	int bg_color = parser.get<int>("bg_color");
	
#ifdef USE_GPU
	htr::CLWrapper clWrapper; // setup OpenCL, the same instance should be used for all following calls to deslantImg
#endif
	// load input image
	cv::Mat img;
	if (inpath == "-")
	{
		char c;
		std::vector<char> data;
		std::cin >> std::noskipws;
		while (std::cin >> c)
			data.push_back(c);
		const cv::Mat rawimg(data);
		img = cv::imdecode(rawimg, cv::IMREAD_GRAYSCALE);
	}
	else
		img = cv::imread(inpath, cv::IMREAD_GRAYSCALE);
#ifdef USE_GPU 
	// deslant on GPU
	const cv::Mat res = htr::deslantImg(img, bg_color, clWrapper);
#else
	// deslant on CPU
	cv::Mat res = htr::deslantImg(img, bg_color, lower_bound, upper_bound);
#endif
	// write result to file
	if (outpath == "-")
	{
		std::vector<unsigned char> data;
		if (cv::imencode(".png", res, data))
		{
			std::string out(data.begin(), data.end());
			std::cout << out;
		}
		else
			return 1;
	}
	else
		cv::imwrite(outpath, res);
	return 0;
}

