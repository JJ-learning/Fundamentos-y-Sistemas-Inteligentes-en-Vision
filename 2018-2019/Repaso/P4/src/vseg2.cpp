// Program created with https://docs.opencv.org/3.4/d3/dbe/tutorial_opening_closing_hats.html
#include <iostream>
#include <exception>
#include <ctype.h>
#include <cmath>
#include <sstream>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/highgui.hpp>		   // imread
#include <opencv2/imgproc/imgproc.hpp> // cvtcolor
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int morpho_option = 0;
int const max_option = 4;
int morpho_ele = 0;
int const max_ele = 2;
int morpho_size = 0;
int const max_kernel_size = 21;

Mat frame;
Mat output;

void morphoFunction(int, void *)
{
	int operation = morpho_option + 2;
	Mat element = getStructuringElement(morpho_ele, Size(2 * morpho_size + 1, 2 * morpho_size + 1), Point(morpho_size, morpho_size));
	morphologyEx(frame, output, operation, element);
	imshow("Video", output);
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{@path |.| path to file\n}"
	"{@out |out.png| path for the output image\n}"

	;

int main(int argc, char *const *argv)
{
	int retCode = EXIT_SUCCESS;

	try
	{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Application name v1.0.0");
		if (parser.has("help"))
		{
			parser.printMessage();
			return 0;
		}

		String input_video = parser.get<String>(0);
		String output = parser.get<String>(1);

		VideoCapture scene(input_video);

		for (;;)
		{
			scene >> frame;
			if (frame.empty())
			{
				cout << "Video ended" << endl;
				exit(-1);
			}

			namedWindow("Video", WINDOW_AUTOSIZE);
			createTrackbar("0.Opening \t1.Closing \t2.Gradient \t3.Top hat \t4.Black hat ", "Video", &morpho_option, max_option, morphoFunction);
			createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", "Video", &morpho_ele, max_ele, morphoFunction);
			createTrackbar("Kernel size:\n 2n +1", "Video", &morpho_size, max_kernel_size, morphoFunction);
			morphoFunction(0, 0);

			if (waitKey(30) >= 0)
				break;
		}
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion: " << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
