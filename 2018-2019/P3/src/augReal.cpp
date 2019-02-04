#include <iostream>
#include <exception>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void runCalibration(vector<vector<Point3f> > objectPoints, vector<vector<Point2f> > imagePoints, Mat frame, String newIntrinsics){
	Mat intrinsic;
    Mat distortionCoef;
    vector<Mat> rotate;
    vector<Mat> translate;
 
    double error = calibrateCamera(objectPoints, imagePoints, frame.size(), intrinsic, distortionCoef, rotate, translate);
    FileStorage file(newIntrinsics, FileStorage::WRITE);
 
    file << "image-width" << frame.size().width;
    file << "image-height" << frame.size().height;
    file << "error" << error;
    file << "camera-matrix" << intrinsic << "distortion-coefficients" << distortionCoef;
    file << "rotation-matrix" << rotate;
    file << "translation-matrix" << translate;
    file.release();
}

void readIntrinsicParams(String intrinsicsFile, Mat &camera, Mat &distortion){
	FileStorage file(intrinsicsFile, FileStorage::READ);
	if(file.isOpened()){
		file["camera-matrix"] >> camera;
		file["distortion-coefficients"] >> distortion;
	}
	
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{@path |.| path to file\n}"
	"{out |out.png| path for the output image\n}"
	"{r || rows of the chessboard\n}"
	"{c || cols of the chessboard\n}"
	"{s || size of the chessboard\n}"
	"{@intrinsic || intrinsincs file\n}"
	"{i || image onto the plane\n}"
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
		String inputFile = parser.get<String>(1);
		String outputFile = parser.get<String>("out");
		int rows = parser.get<int>("r");
		int cols = parser.get<int>("c");
		int size = parser.get<int>("s");
		String intrinsicsFile = parser.get<String>(0);
		String inputFile2 = parser.get<String>("i");
		
		// Variables needed
		Mat frame;
		Mat grayFrame;
		VideoCapture scene(inputFile);
		VideoCapture scene2;
		if(parser.has("i")){
			scene2.open(inputFile2);
		}
		
		bool patternIsFound;// Keeps if a pattern is found
		
		if(!scene.isOpened()){
			printf("No video data\n");
			return -1;
		}

		if(!parser.check()){
			parser.printErrors();
			return 0;
		}

		while(scene.read(frame)){
			if(frame.empty()){
				cout<<"Error! Empty frame"<<endl;
				break;
			}

			vector<vector<Point3f> > objectPoints;// Keeps the position of the object
			vector<vector<Point2f> > imagePoints;// Keeps 
			vector<Point3f> object;
			vector<Point2f> corners;
	
			// We convert it to gray and get the board size
			Size boardSize(rows-1, cols-1);

			// Keeps the position of each vertex of the world coordinates
			for (int i = 0; i < cols - 1; ++i){
				for(int j = 0; j < rows - 1;j++){
					object.push_back(Point3f(i*size, j*size, 0));
				}
			}
			
			// Look for imagePoints(corners)
			patternIsFound = findChessboardCorners(frame, boardSize, corners);
			if(patternIsFound == true){
				// Change the color of the frame
				cvtColor(frame, grayFrame, CV_BGR2GRAY);
				
				// Enhance the imagePoints
				cornerSubPix(grayFrame, corners, Size(11,11), Size(-1,-1),TermCriteria());
				imagePoints.push_back(corners);
				objectPoints.push_back(object);

				// Create the new intrinsics file
				size = size - 1;
				if(!size){
					runCalibration(objectPoints, imagePoints, frame, intrinsicsFile);
				}
				
				Mat camera, distortion, rotations, translations;
				// Read the intrinscis parameters
				readIntrinsicParams(intrinsicsFile, camera, distortion);
				
				// We find the object pose from the points
				solvePnP(Mat(object), Mat(corners), camera, distortion, rotations, translations);
				
				// Option -i
				if(parser.has("i") == false){
					vector<Point3f> projectedPoints;
					// We keep the points of the projected object
					projectedPoints.push_back(Point3f((rows/2) * size, (cols/2) * size, 0));
					projectedPoints.push_back(Point3f(((rows/2)+1) * size, (cols/2) * size, 0));
					projectedPoints.push_back(Point3f((rows/2) * size, ((cols/2) + 1) * size, 0));
					projectedPoints.push_back(Point3f((rows/2) * size, (cols/2) * size, -size));
					
					vector<Point2f> output;
					// Project the 3D points onto an image's plane
					projectPoints(Mat(projectedPoints), Mat(rotations), Mat(translations), camera, distortion, output);
					
					// Colors RGB to draw the object
					Scalar red(255, 0, 0);
					Scalar green(0, 255, 0);
					Scalar blue(0, 0, 255);
					
					// Draw the final object connecting the points (output[0] with output[1], ...)
					line(frame, output[0], output[1], red, 3);
					line(frame, output[0], output[2], green, 3);
					line(frame, output[0], output[3], blue, 3);
				}else{
					Mat frame2;
					if(scene2.read(frame2) == false){
						scene2.release();
						scene2.open(inputFile2);
						scene2.read(frame2);
					}

					vector<Point2f> imagePoints2;
					vector<Point3f> object2;

					object2.push_back(Point3f((rows-1)*size,(cols-1)*size, 0));
                    object2.push_back(Point3f(0, 0, 0));
                    object2.push_back(Point3f((rows-1)*size, 0, 0));
                    object2.push_back(Point3f(0, (cols-1)*size, 0));

					vector<Point2f> output;
					// Project the 3D points onto an image's plane
					projectPoints(Mat(object2), Mat(rotations), Mat(translations), camera, distortion, output);

					imagePoints2.push_back(Point2f(0,0));   
                    imagePoints2.push_back(Point2f(frame2.cols,frame2.rows));
                    imagePoints2.push_back(Point2f(frame2.cols,0)); 
                    imagePoints2.push_back(Point2f(0,frame2.rows));

					// We get the size of the perspective
					Size perspectiveSize = Size((int) scene.get(CV_CAP_PROP_FRAME_WIDTH), (int) scene.get(CV_CAP_PROP_FRAME_HEIGHT));
					// Get the transform
					Mat perspective = getPerspectiveTransform(imagePoints2, output);
					// Write the frame into the scene
					warpPerspective(frame2, frame, perspective, perspectiveSize, INTER_LINEAR, BORDER_TRANSPARENT);
				}
			}
			
			// Show the final object
			imshow("Output", frame);
			if(waitKey(5)>=0){
				break;
			}
		
		}
	}catch(exception& e){
		cerr << "Capturada excepcion" << e.what() <<endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
