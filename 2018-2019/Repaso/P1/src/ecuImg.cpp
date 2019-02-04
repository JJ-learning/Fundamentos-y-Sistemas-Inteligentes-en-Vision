#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
/*
	First, we have to read the image in B&W. Then we have to:
		1. Calculate the histogram
		2. Nomalizate the histogram
		3. Get equalization
	Note: If the radius is greatest than 0, we calculate the histogram of the image but considering this time, the radius.
*/
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

vector<int> calculateHistogram(Mat &input, Mat mask, bool maskoption)
{
	// To Do
	vector<int> hist(256, 0);
	for (int i = 0; i < input.rows; i++)
	{
		uchar *ptr = input.ptr<uchar>(i);
		for (int j = 0; j < input.cols; j++)
		{
			if (mask.at<uchar>(i, j) != 0 && maskoption == false)
			{
				hist[ptr[j]] += 1;
			}
			else if (maskoption == true)
			{
				hist[ptr[j]] += 1;
			}
			else if (mask.empty())
			{
				hist[ptr[j]] += 1;
			}
		}
	}
	return hist;
}

vector<int> normalizarHistograma(vector<int> histograma)
{
	// To Do
	vector<int> aux(256, 0);
	aux[0] = histograma[0];
	for (int i = 1; i < histograma.size(); i++)
	{
		aux[i] = aux[i - 1] + histograma[i];
	}
	for (int i = 0; i < histograma.size(); i++)
	{
		aux[i] = (int)(((float)aux[i] / aux[histograma.size() - 1]) * histograma.size() - 1);
	}
	return aux;
}

Mat getEqualization(Mat input, vector<int> histograma, Mat mask)
{
	// To Do
	Mat output = input.clone();
	for (int i = 0; i < input.rows; i++)
	{
		uchar *ptr = output.ptr<uchar>(i);
		for (int j = 0; j < input.cols; j++)
		{
			if (mask.at<uchar>(i, j) != 0)
			{
				ptr[j] = histograma[ptr[j]];
			}
			else if (mask.empty())
			{
				ptr[j] = histograma[ptr[j]];
			}
		}
	}
	return output;
}

Mat getEqualizationRadius(Mat input, int radius, Mat mask)
{
	// To Do
	Mat aux;
	Mat output = input.clone();
	for (int i = radius; i < input.rows - radius; i++)
	{
		uchar *ptr = output.ptr<uchar>(i);
		for (int j = radius; j < input.cols - radius; j++)
		{
			if (mask.at<uchar>(i, j) != 0)
			{
				double pos = input.at<uchar>(i, j);
				aux = input(Rect(j - radius, i - radius, 2 * radius + 1, 2 * radius + 1));
				vector<int> hist = calculateHistogram(aux, mask, true);
				vector<int> hist_norm = normalizarHistograma(hist);
				ptr[j] = hist_norm[pos];
			}
		}
	}
	return output;
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{path |.| path to file\n}"
	"{r |0| Radius of the window\n}"
	"{out |out.png| path for the output image\n}"
	"{mask|| mask used in the equalization}"
	"{hsv|| option to calculate the histogram with a color image\n}"
	"{b || option to calculate bipartional equalization\n}";

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
		String inputFile = parser.get<String>("path");
		String outputFile = parser.get<String>("out");
		String maskFile = parser.get<String>("mask");
		int r = parser.get<double>("r");

		bool isHSV = parser.has("hsv");
		bool bipartional = parser.has("b");

		if (!parser.check())
		{
			parser.printErrors();
			return 0;
		}

		// Common variables
		Mat inputImage = imread(inputFile, 0);
		Mat outputImage;
		Mat mask = imread(maskFile, 0);

		vector<int> histograma;
		vector<int> histograma_normalizado;

		// Check the image is loaded successfully
		checkImage(inputImage);

		// Show input image
		namedWindow("Input image", WINDOW_AUTOSIZE);
		imshow("Input image", inputImage);

		// Check the Radius
		if (r > inputImage.rows / 2 || r > inputImage.cols / 2)
		{
			cout << "Error! The Radius value is invalid.\n\tExiting program..." << endl;
			exit(-1);
		}

		// BEGINING OF THE PROGRAM
		if (r == 0)
		{
			histograma = calculateHistogram(inputImage, mask, false);

			histograma_normalizado = normalizarHistograma(histograma);

			outputImage = getEqualization(inputImage, histograma_normalizado, mask);
		}
		else
		{
			outputImage = getEqualizationRadius(inputImage, r, mask);
		}

		// Show output image
		namedWindow("Output image", WINDOW_AUTOSIZE);
		imshow("Output image", outputImage);
		imwrite(outputFile, outputImage);

		// Wait for any key pressed
		waitKey(0);
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion" << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}