#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PI 3.14159265359879323846

using namespace cv;
using namespace std;

Mat filter;
Mat inputImage;
Mat outputImage;
Mat filteredImage;
int rTrackbar;
int gainTrackbar;
string outputFile;
int mode;

Mat createBoxFilter(int r){
	Mat filter(r, r, CV_32FC1);


	for (int i = 0; i < filter.rows; ++i)
	{
	 	for (int j = 0; j < filter.cols; ++j)
	 	{
	 		filter.at<float>(i,j) = 1/pow(r,2);
	 	}
	}
	return filter;
}

float gaussFunction(int i, int j){
	float exponent = -((pow(i, 2) + pow(j, 2))/2);
	float aux = pow( exp(1), exponent);
	float aux2 = 1/(2*PI);
	float aux3 = aux * aux2;

	return aux3;
}

Mat createGaussianFilter(int r){
	Mat filter(r, r, CV_32FC1);
	int middle = r/2;
	for (int i = 0; i < filter.rows; ++i)
	{
		for (int j = 0; j < filter.cols; ++j)
		{
			filter.at<float>(i,j) = gaussFunction(i - middle, j - middle);
		}
	}

	return filter;
}

// g(i,j) = sum(f(i+k,j+l)*h(k,l))
void applyFilter(Mat &inputImage, Mat &filter, Mat &filteredImage){
	// We read the input image
	for (int i = 0; i < inputImage.rows; ++i)
	{
		for (int j = 0; j < inputImage.cols; ++j)
		{
			float sum = 0.0;
			int iFilter = 0, jFilter = 0;
			if (i == 0 || i == inputImage.rows-1 || j == 0 || j == inputImage.cols-1)
			{
				sum = inputImage.at<float>(i,j);
			}else{
				do{
					do{
						sum += inputImage.at<float>(i + iFilter, j + jFilter)*filter.at<float>(iFilter, jFilter);
						jFilter++;
					}while(jFilter < filter.cols);
					iFilter++;
					jFilter = 0;
				}while(iFilter < filter.rows);
			}
			filteredImage.at<float>(i,j) = sum;
		}
	}
}

void applyGain(Mat &image, int gain){
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			image.at<float>(i,j) *= gain;
		}
	}
}

void convolve(Mat inputImage, Mat filter, Mat &outputImage, float gain){

	Mat filteredImage(inputImage.rows, inputImage.cols, CV_32FC1);

	applyFilter(inputImage, filter, filteredImage);
			
	applyGain(inputImage, gain+1);
	applyGain(filteredImage, gain);

	for (int i = 0; i < inputImage.rows; ++i)
	{
		for (int j = 0; j < inputImage.cols; ++j)
		{
			outputImage.at<float>(i,j) = inputImage.at<float>(i,j) - filteredImage.at<float>(i,j);
			if(outputImage.at<float>(i,j) > 255){
				outputImage.at<float>(i,j) = 255;
			}
		}
	}
}

int checkImage(Mat &inputImage){
	if( ! inputImage.data){
		cout << "Image not correctly loaded" << endl;
		return 0;
	}else{
		cout << "Image correctly loaded" << endl;
		cout<<"Image's channels: "<<inputImage.channels()<<endl;
	}	
	return 0;
}

void functionSlider(int , void* ){
	rTrackbar = 2*rTrackbar+1;

	if(mode == 0){
		filter = createBoxFilter(rTrackbar);
		convolve(inputImage, filter, outputImage, gainTrackbar);
	}else if(mode ==1){
		filter = createGaussianFilter(rTrackbar);
		convolve(inputImage, filter, outputImage, gainTrackbar);
	}

	imshow("Output", outputImage);	
}


const String keys =
	"{help h usage ? || print this message\n}"
	"{@path |.| path to file\n}"
	"{out |out.png| path for the output image\n}"
	"{r |1| filter size\n}"
	"{g |1.0| gain\n}"
	"{f |0| Filter mode}"
	"{i | | Interactive mode}"
	"{v | | hsv mode}"
	;

int main(int argc, char* const *argv){
	int retCode = EXIT_SUCCESS;

	try{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Practica 2");
		if(parser.has("help")){
			parser.printMessage();
			return 0;
		}

		// Command variables
		String inputFile = parser.get<String>(0);
		int r = parser.get<int>("r");
		int gain = parser.get<int>("g");
		mode = parser.get<int>("f");
		outputFile = parser.get<String>("out");

		if(!parser.check()){
			parser.printErrors();
			return 0;
		}

		Mat layers[3];

		// New variables
		inputImage = imread(inputFile);
				
		outputImage = Mat(inputImage.rows, inputImage.cols, CV_32FC1);

		checkImage(inputImage);
		
		inputImage.convertTo(inputImage, CV_32FC1);

		// Check if the gain is between the range
		if(gain < 0.0 || gain > 10.0){
			cout<<"Error! The gain should be between 0.0 and 10.0. \nExiting program..."<<endl;
			return -1;
		}

		// Check if the ratio is between the range
		if(r > 1){
			cout<<"Error! the size of the filter is greater than allowed. \nExiting program..."<<endl;
			return -1;
		}
		r = 2*r + 1;
		
		if(parser.has("i")){
			
			namedWindow("Output", WINDOW_AUTOSIZE);
			createTrackbar("Ratio", "Output", &rTrackbar, 1, functionSlider);
			createTrackbar("Gain", "Output", &gainTrackbar, 1, functionSlider);

			imshow(inputFile, inputImage);
			waitKey(0);
			
		}else if(parser.has("v")){
			cvtColor(inputImage, inputImage, CV_BGR2HSV);
			split(inputImage, layers);
			layers[2].copyTo(inputImage);
			
			if(mode == 0){
				filter = createBoxFilter(r);
				convolve(inputImage, filter, outputImage, gain);
			}else if(mode ==1){
				filter = createGaussianFilter(r);
				convolve(inputImage, filter, outputImage, gain);
			}

			outputImage.copyTo(layers[2]);
			merge(layers, 3, outputImage);
			cvtColor(outputImage, outputImage, CV_HSV2BGR);

			imwrite(outputFile, outputImage);
		}
		else{
			cvtColor(inputImage, inputImage, CV_BGR2GRAY);			
			if(mode == 0){
				filter = createBoxFilter(r);
				convolve(inputImage, filter, outputImage, gain);
			}else if(mode ==1){
				filter = createGaussianFilter(r);
				convolve(inputImage, filter, outputImage, gain);
			}

			imwrite(outputFile, outputImage);
		}

	}catch(exception& e){
		cerr << "Capturada excepcion" << e.what() <<endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
