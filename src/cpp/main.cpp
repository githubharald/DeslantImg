#ifdef USE_GPU
#include "DeslantImgGPU.hpp"
#else
#include "DeslantImgCPU.hpp"
#endif
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <list>


int main(int argc, const char *argv[])
{
	const cv::String opts =
		"{help h usage ? |      | print this message   }"
		"{data           |data  | directory to read the input images from }"
		"{dataout        |.     | directory to write the output images to }"
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
	cv::String inpath = parser.get<cv::String>("data");
	cv::String outpath = parser.get<cv::String>("dataout");
	float lower_bound = parser.get<float>("lower_bound");
	float upper_bound = parser.get<float>("upper_bound");
	int bg_color = parser.get<int>("bg_color");
	
	std::list<cv::String> files;
	std::vector<cv::String> extfiles;
	std::vector<cv::String> extensions { ".png", ".jpg", ".bmp" };
	for (const cv::String& ext : extensions)
	{
		cv::glob(inpath + "/*" + ext, extfiles);
		for (const cv::String& file : extfiles)
			files.push_back(file);
	}
#ifdef USE_GPU
	htr::CLWrapper clWrapper; // setup OpenCL, the same instance should be used for all following calls to deslantImg
#endif
	for (const cv::String & file : files)
	{
		std::cout << "Reading '" << file << "'" << std::endl;
		// load input image
		const cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
#ifdef USE_GPU 
		// deslant on GPU
		const cv::Mat res = htr::deslantImg(img, bg_color, clWrapper);
#else
		// deslant on CPU
		cv::Mat res = htr::deslantImg(img, bg_color, lower_bound, upper_bound);
#endif
		// write result to file
		cv::String out = outpath + "/" + file.substr(file.find_last_of("/\\") + 1);
		std::cout << "Writing '" << out << "'" << std::endl;
		cv::imwrite(out, res);
	}
	return 0;
}

