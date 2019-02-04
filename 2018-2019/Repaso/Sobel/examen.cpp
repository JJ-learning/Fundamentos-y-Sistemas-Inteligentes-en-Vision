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

Mat sumatory(Mat frame1, Mat frame2){
    Mat aux = Mat::zeros(frame1.rows, frame1.cols, frame1.type());
    aux = frame1 + frame2;
    return aux;
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


        // VideoCapture scene(inputFile);
        // Mat frame;
        // Mat avg;
        // int numbre_frame = 0;

		// while(scene.grab()){
		// 	scene.retrieve(frame);
		// 	frame.convertTo(frame, CV_32S);
		// 	if(frame.empty()){
		// 		break;
		// 	}
		// 	if(numbre_frame == 0){
		// 		avg = frame.clone();
		// 	}else{
		// 		avg = sumatory(avg, frame);
		// 	}
		// 	numbre_frame++;
			

		// }
		// cout<<avg<<endl;
		// imshow("avg", avg/numbre_frame);
		// waitKey(0);

		Mat input = imread(inputFile, 0);
		input.convertTo(input, CV_32F);
		Mat input2 = input.clone();
		Mat input3 = input.clone();

		Mat result = Mat(input.rows, input.cols, CV_32F);

		for(int i=0; i<input.rows; i++){
			float *ptr_result = result.ptr<float>(i);
			float *ptr = input.ptr<float>(i);
			float *ptr2 = input2.ptr<float>(i);
			float *ptr3 = input3.ptr<float>(i);
			for(int j=0; j<input.cols; j++, ptr++, ptr2++, ptr3++, ptr_result++){
				*ptr_result = *ptr2 + *ptr + *ptr3;
			}
		}

		cout<<result<<endl;
        return 1;
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion" << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}