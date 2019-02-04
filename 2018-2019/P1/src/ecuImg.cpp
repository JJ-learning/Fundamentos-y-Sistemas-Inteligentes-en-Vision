#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

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

vector<int> calculateHistogram(Mat &input){
	vector<int> histograma(256,0);
    for(int i=0; i<input.rows; i++){
        uchar *ptr = input.ptr<uchar>(i);
        for(int j=0; j<input.cols; j++){
            histograma[ptr[j]] += 1;
        }
    }
	return histograma;
}

vector<int> normalizarHistograma(vector<int> histograma){
    vector<int> aux(256,0);
    aux[0] = histograma[0];
    for(int i=1; i<256; i++){
        aux[i] = aux[i-1] + histograma[i];
    }
    for(int i=0; i<256; i++){
        aux[i] = (int) (((float) aux[i]/aux[255]) * 255);
    }
    return aux;
}

Mat getEqualization(Mat inputImage, vector<int> histograma){
	Mat output = inputImage.clone();
	for(int i=0; i<inputImage.rows; i++){
        uchar *ptr = output.ptr<uchar>(i);
        for(int j=0; j<inputImage.cols; j++){
            ptr[j] = histograma[ptr[j]];
        }
    }
	return output;
}

Mat getEqualizationRatio(Mat input, int ratio){
    Mat auxMat;
    Mat output = input.clone();

    for(int i=ratio; i<input.rows-ratio; i++){
        for(int j=ratio; j<input.cols-ratio; j++){
            double pos = input.at<uchar>(i,j);
            auxMat = input(Rect(j-ratio, i-ratio, 2*ratio+1, 2*ratio+1));
            vector<int> histograma = calculateHistogram(auxMat);
            vector<int> histograma_normalizado = normalizarHistograma(histograma);
            output.at<uchar>(i,j) = histograma_normalizado[pos];
        }
    }
	return output;
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{path |.| path to file\n}"
	"{r |0| ratio of the window\n}"
	"{out |out.png| path for the output image\n}"
	"{mask|| mask used in the equalization}"
	"{hsv|| option to calculate the histogram with a color image\n}"
	"{b || option to calculate bipartional equalization\n}"
	;


int main(int argc, char const *argv[])
{

	int retCode = EXIT_SUCCESS;

	try{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Practica 1");
		if(parser.has("help")){
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

		if(!parser.check()){
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

		// Check the ratio
		if (r > inputImage.rows/2 || r > inputImage.cols/2)
		{
			cout<<"Error! The ratio value is invalid.\n\tExiting program..."<<endl;
			exit(-1);
		}

		// BEGINING OF THE PROGRAM
		if(r == 0){
			histograma = calculateHistogram(inputImage);

			histograma_normalizado = normalizarHistograma(histograma);

			outputImage = getEqualization(inputImage, histograma_normalizado);
		}else{
			outputImage = getEqualizationRatio(inputImage, r);
		}

		// Show output image
		namedWindow("Output image", WINDOW_AUTOSIZE);
		imshow("Output image", outputImage);
		imwrite(outputFile, outputImage);

		// Wait for any key pressed
		waitKey(0);

	}catch(exception& e){
		cerr << "Capturada excepcion" << e.what() <<endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}