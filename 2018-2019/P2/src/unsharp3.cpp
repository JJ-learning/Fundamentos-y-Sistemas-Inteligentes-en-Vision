#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <assert.h>

#include <cmath>

using namespace cv;
using namespace std;


#define PI 3.1415926535897 // pi

Mat createBoxFilter(int r){
	Mat filter (2*r+1, 2*r+1, CV_32FC1);
	filter = 1.0/pow(2*r+1, 2);

    return filter;
}


void convolve(const Mat &input, const Mat &filter, Mat &output){
	
    assert(input.type() == CV_32FC1 && filter.type() == CV_32FC1);
	assert(input.cols == output.cols && input.rows == output.rows);

	int diameter = filter.rows;
	int radius = diameter/2;

	for(int i=0; i < (input.rows - diameter); i++){
		float *ptr_out = output.ptr<float>(i + radius);
		for(int j=0; j < (input.cols - diameter); j++){
			Mat window = input(Rect(j,i, diameter, diameter));
			// Apply filter
			for(int k=0; k < diameter; k++){
				float const *ptr_filter = filter.ptr<const float>(k);
				float const *ptr_window = window.ptr<const float>(diameter - k);
				for(int l=0; l < diameter; l++, ptr_filter++){
					ptr_out[j + radius] += *ptr_filter * ptr_window[diameter - l];
				}
			}
		}
	}	
}

int getRadius(float r, int rows, int cols){
    int radius;
	if(rows < cols){
		radius = (r*(((rows-1)/2)-1))+1;
	}else{
		radius = (r*(((cols-1)/2)-1))+1;
	}

	return radius;
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
    Mat inputImage;
    Mat filter;
    Mat outputImage;
	Mat convolvedImage;
	
	try{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Practica 2");
		if(parser.has("help")){
			parser.printMessage();
			return 0;
		}

		// Command variables
		String inputFile = parser.get<String>(0);
		float r = parser.get<float>("r");
		float gain = parser.get<float>("g");
		int mode = parser.get<int>("f");
		String outputFile = parser.get<String>("out");

		if(!parser.check()){
			parser.printErrors();
			return 0;
		}

		// New variables
		inputImage = imread(inputFile, 0);
		

		// We check if the image has been opened
		checkImage(inputImage);

		// Check if the gain is between the range
		if(gain < 0.0 || gain > 10.0){
			cout<<"Error! The gain should be between 0.0 and 10.0. \nExiting program..."<<endl;
			return -1;
		}

		// Check if the Radius is between the range
		if(r > (min(inputImage.cols, inputImage.rows)/2)){
			cout<<"Error! the size of the filter is greater than allowed. \nExiting program..."<<endl;
			return -1;
		}

		// BEGINING OF THE PROGRAM
		imshow("Input", inputImage);

        // We convert the image to CV_32FC1 type
		inputImage.convertTo(inputImage, CV_32F, 1.0/255.0, 0.0);
        outputImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_32FC1);
		convolvedImage = Mat::zeros(inputImage.rows, inputImage.cols, CV_32FC1);

        // First, we get the filters Radius according to the image dimensions
        int radius = getRadius(r, inputImage.rows, inputImage.cols);

        if(mode == 0){
            filter = createBoxFilter(radius);
        }

        // We do the convolve process
        convolve(inputImage, filter, convolvedImage);
		// cout<<outputImage;
        imshow("Convolved", convolvedImage);
		
		outputImage = ((gain+1)*inputImage - (gain * convolvedImage));

		outputImage.convertTo(outputImage, CV_8U, 255.0/1.0, 0.0);

        imshow("Ouput", outputImage);
        waitKey(0);

        imwrite("out.png", outputImage);

	}catch(exception& e){
		cerr << "Capturada excepcion" << e.what() <<endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
