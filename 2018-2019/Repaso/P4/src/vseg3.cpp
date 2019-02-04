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

int shape = 0;
const int max_shape = 2;
int kernel_size = 0;
int const max_kernel_size = 21;
int umbral = 10;
int const max_umbral = 255;

Mat frame, frame2;
Mat opening;
Mat closing;
Mat output;

Mat calculateFrameDifference(Mat frame, Mat frame2, int umbral, int kernel_size)
{
	Mat diff;

	// Get the difference between the two frames
	absdiff(frame, frame2, diff);
	// Apply threshold
	diff = diff > umbral;

	if (kernel_size == 0)
	{
		return diff; // Any operation is needed
	}
	Mat aux;
	Mat output_function, opening, closing;

	// Get the structure and apply the opening and closing operators
	Mat shape = getStructuringElement(MORPH_ELLIPSE, Size(2 * kernel_size + 1, 2 * kernel_size + 1), Point(kernel_size, kernel_size));
	morphologyEx(diff, opening, MORPH_OPEN, shape, Point(kernel_size, kernel_size), 1, BORDER_DEFAULT);
	morphologyEx(opening, closing, MORPH_CLOSE, shape, Point(kernel_size, kernel_size), 1, BORDER_DEFAULT);

	// frame & closing
	output_function = frame & closing;

	return output_function;
}

void addFunction(int, void *)
{
	output = calculateFrameDifference(frame, frame2, umbral, kernel_size);

	imshow("Video", output);
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{@path |.| path to file\n}"
	"{@out |out.png| path for the output image\n}";

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
		String output_video = parser.get<String>(1);

		int cont = 0;

		VideoCapture scene(input_video);

		while (scene.grab())
		{
			scene.retrieve(frame);
			if (frame.empty())
			{
				cout << "Programa acabado" << endl;
				exit(-1);
			}
			if (cont != 0 && scene.grab())
			{
				scene.retrieve(frame2);
				namedWindow("Video", WINDOW_AUTOSIZE);
				createTrackbar("Kernel size:", "Video", &kernel_size, max_kernel_size, addFunction);
				createTrackbar("Umbral: ", "Video", &umbral, max_umbral, addFunction);
				createTrackbar("Shape: ", "Video", &shape, max_shape, addFunction);
				addFunction(0, 0);

				if (waitKey(30) >= 0)
					break;
			}
			cont++;
		}
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion: " << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
