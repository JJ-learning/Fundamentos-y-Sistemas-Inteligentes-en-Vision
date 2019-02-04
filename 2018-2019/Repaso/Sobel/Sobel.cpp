#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int checkImage(Mat &inputImage)
{
	if (!inputImage.data)
	{
		cout << "Image not correctly loaded" << endl;
		return 0;
	}
	else
	{
		cout << "Image correctly loaded" << endl;
		cout << "Image's channels: " << inputImage.channels() << endl;
	}
	return 0;
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{@image |.| path to file\n}"
	"{mode | 0 | mode of the operator used\n}";

int main(int argc, char const *argv[])
{

	int retCode = EXIT_SUCCESS;

	try
	{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Practica 1");
		if (parser.has("help"))
		{
			parser.printMessage();
			return 0;
		}

		//Command variables
		String inputFile = parser.get<String>(0);
		int mode = parser.get<int>("mode");

		if (!parser.check())
		{
			parser.printErrors();
			return 0;
		}
		// Needed variables
		Mat image;
		Mat src;
		Mat src_gray;
		Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
		Mat output;
		String name;

		// Read the image
		image = imread(inputFile);

		// Reduce noise
		GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

		// Convert to gray
		cvtColor(src, src_gray, CV_BGR2GRAY);
		imshow("input", src);
		if (mode == 0)
		{
			// To do: Apply the Sobel operator
			Sobel(src_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
			Sobel(src_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

			// Do convertScaleAbs
			convertScaleAbs(grad_x, abs_grad_x);
			convertScaleAbs(grad_y, abs_grad_y);

			// Add weight
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output);
		}
		else if (mode == 1)
		{
			// Laplacian operator
			Laplacian(src_gray, src, CV_16S, 3, 1, 0, BORDER_DEFAULT);
			convertScaleAbs(src, output);
			name = "Laplacian detector";
		}

		imshow(name, output);
		waitKey(0);
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion" << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}