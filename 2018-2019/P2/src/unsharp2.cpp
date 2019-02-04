#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

using namespace cv;
using namespace std;

Mat inputImage;
Mat filter;
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
	 		filter.at<float>(i,j) = (float) 1/pow(r,2);
	 	}
	}
	return filter;
}

float gaussFunction(int i, int j, int r){
	float exponent = -((pow(i, 2) + pow(j, 2))/2*r);
	float aux = pow(M_E, exponent ) * (1/(2 * M_PI * pow(r, 2)));

	return aux;
}

Mat createGaussianFilter(int r){
	Mat filter(r, r, CV_32FC1);
	int middle = r/2;
	for (int i = 0; i < filter.rows; ++i)
	{
		for (int j = 0; j < filter.cols; ++j)
		{
			filter.at<float>(i,j) = gaussFunction(i - middle, j - middle, r);
		}
	}

	return filter;
}

void applyFilter(Mat inputImage, Mat filteredImage, Mat &outputImage){
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

void applyGain(Mat &image, int gain){
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			image.at<float>(i,j) = image.at<float>(i,j) * gain;
		}
	}
}

void convolve(Mat inputImage, Mat filter, Mat &outputImage){
    float sum;
    int size = filter.rows;

    for(int i = 0; i < inputImage.rows; i++)
    {
        float *ptr = outputImage.ptr<float>(i);
        
        for(int j = 0; j < inputImage.cols; j++)
        {
            if ((size/2)+1 < i && ((size/2)+1) < j && (inputImage.rows-(size/2)) > i && (inputImage.cols-(size/2) > j)) {
                Mat aux(inputImage, Rect(j-(size/2)-1, i-(size/2), size, size));
                
                
                for(int k = 0; k < filter.rows; k++)
                {
                    float *ptraux = aux.ptr<float>(k);
                    float *prtfilter = filter.ptr<float>(k);

                    
                    for(int l = 0; l < filter.cols; l++)
                    {
                        sum +=  (ptraux[l] * prtfilter[l]);
                    }
                    
                }
                ptr[j] = sum;
                sum = 0.0;
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

int getMinimum(Mat inputImage){
    if(inputImage.rows > inputImage.cols){
        return inputImage.cols/2;
    }else{
        return inputImage.rows/2;
    }
}

int sizeWindow(float r, int minimum){
    int aux;
    if(r != 0){
        aux = (int)(r * minimum + 1);
        if(aux < 3){
            return 3;
        }else{
            if((aux%2) != 0){
                return aux;
            }else{
                return (aux-1);
            }
        }
    }else{
        return 3;
    }
}

void functionSlider(int , void* ){
    int minimum = getMinimum(inputImage);
	if(mode == 0){
        int size = sizeWindow(rTrackbar, minimum);
        filter = createBoxFilter(size);
    }else if(mode ==1){
        int size = sizeWindow(rTrackbar, minimum);
        filter = createGaussianFilter(size);
    }

    convolve(inputImage, filter, filteredImage);

	applyGain(inputImage, gainTrackbar+1);
	applyGain(filteredImage, gainTrackbar);

	applyFilter(inputImage, filteredImage, outputImage);

	imwrite(outputFile, outputImage);	
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
		float r = parser.get<float>("r");
		float gain = parser.get<float>("g");
		mode = parser.get<int>("f");
		outputFile = parser.get<String>("out");

		if(!parser.check()){
			parser.printErrors();
			return 0;
		}

		Mat layers[3];

		// New variables
        int minimum = 0;
		inputImage = imread(inputFile);
		outputImage = Mat(inputImage.rows, inputImage.cols, CV_32FC1);

		// We check if the image has been opened
		checkImage(inputImage);

		// We convert the image to CV_32FC1 type
		inputImage.convertTo(inputImage, CV_32FC1);

		// Check if the gain is between the range
		if(gain < 0.0 || gain > 10.0){
			cout<<"Error! The gain should be between 0.0 and 10.0. \nExiting program..."<<endl;
			return -1;
		}

		// Check if the ratio is between the range
		if(r > (min(inputImage.cols, inputImage.rows)/2)){
			cout<<"Error! the size of the filter is greater than allowed. \nExiting program..."<<endl;
			return -1;
		}
		
		if(parser.has("i")){
            rTrackbar = r;
            gainTrackbar = gain;

			namedWindow("Output", WINDOW_AUTOSIZE);
			createTrackbar("Ratio", "Output", &rTrackbar, 1, functionSlider);
			createTrackbar("Gain", "Output", &gainTrackbar, 1, functionSlider);

			imshow(inputFile, inputImage);
			waitKey(0);
			
		}else{
            minimum = getMinimum(inputImage);
            
            if(parser.has("v")){
                cvtColor(inputImage, inputImage, CV_BGR2HSV);
				filteredImage = Mat(inputImage.rows, inputImage.cols, CV_32FC1);
				

                split(inputImage, layers);
                layers[2].copyTo(inputImage);
			
                if(mode == 0){
                    int size = sizeWindow(r, minimum);
                    filter = createBoxFilter(size);
                }else if(mode ==1){
                    int size = sizeWindow(r, minimum);
                    filter = createGaussianFilter(size);
                }

                convolve(inputImage, filter, outputImage);

				applyGain(inputImage, gain+1);
				applyGain(outputImage, gain);

				applyFilter(inputImage, filter, outputImage);

                outputImage.copyTo(layers[2]);
                merge(layers, 3, outputImage);
                cvtColor(outputImage, outputImage, CV_HSV2BGR);

                imwrite(outputFile, outputImage);
            }
            else{
                cvtColor(inputImage, inputImage, CV_BGR2GRAY);			
				filteredImage = Mat(inputImage.rows, inputImage.cols, CV_32FC1);

                if(mode == 0){
                    int size = sizeWindow(r, minimum);
                    filter = createBoxFilter(size);
                }else if(mode ==1){
                    int size = sizeWindow(r, minimum);
                    filter = createGaussianFilter(size);
                }

                convolve(inputImage, filter, filteredImage);

				applyGain(inputImage, gain+1);
				applyGain(filteredImage, gain);

				applyFilter(inputImage, filteredImage, outputImage);

                imwrite(outputFile, outputImage);
            }
        }

	}catch(exception& e){
		cerr << "Capturada excepcion" << e.what() <<endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
