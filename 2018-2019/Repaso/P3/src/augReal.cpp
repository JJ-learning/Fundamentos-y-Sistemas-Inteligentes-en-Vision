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

void runCalibration(vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> imagePoints, Mat frame, String newIntrinsics)
{
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

void readIntrinsicParams(String intrinsicsFile, Mat &camera, Mat &distortion)
{
	FileStorage file(intrinsicsFile, FileStorage::READ);
	if (file.isOpened())
	{
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
	"{i || image onto the plane\n}";

int main(int argc, char *const *argv)
{
	int retCode = EXIT_SUCCESS;

	try
	{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Practica 2");
		if (parser.has("help"))
		{
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
		if (parser.has("i"))
		{
			scene2.open(inputFile2);
		}

		bool patternIsFound; // Keeps if a pattern is found

		if (!scene.isOpened())
		{
			printf("No video data\n");
			return -1;
		}

		if (!parser.check())
		{
			parser.printErrors();
			return 0;
		}

		while (scene.read(frame))
		{
			if (frame.empty())
			{
				cout << "Error! Empty frame" << endl;
				break;
			}
			// TODO
			vector<Point3f> object;
			vector<Point2f> corners;
			Size boardSize(rows - 1, cols - 1);
			for (int i = 0; i < cols + 1; i++)
			{
				for (int j = 0; j < rows + 1; j++)
				{
					object.push_back(Point3f(j * size, i * size, 0));
				}
			}

			bool patterFound = findChessboardCorners(frame, boardSize, corners);
			if (patterFound == true)
			{
				cvtColor(frame, grayFrame, CV_BGR2GRAY);
				cornerSubPix(grayFrame, corners, Size(11, 11), Size(-1, -1), TermCriteria());
				Mat camera, distortion, rotations, translations;
				readIntrinsicParams(intrinsicsFile, camera, distortion);

				solvePnP(Mat(object), Mat(corners), camera, distortion, rotations, translations);

				vector<Point3f> projectedPoints;

				projectedPoints.push_back(Point3f((rows / 2) * size, (cols / 2) * size), 0);
				projectedPoints.push_back(Point3f(((rows / 2) + 1) * size, (cols / 2) * size), 0);
			}

			// Show the final object
			imshow("Output", frame);
			if (waitKey(5) >= 0)
			{
				break;
			}
		}
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion" << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
