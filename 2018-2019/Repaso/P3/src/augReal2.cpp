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

void drawCoordenateAxis(Mat &image, int size, const Mat &cameraMatrix, const Mat &distCoef, const Mat &rvec, const Mat &tvec)
{
    vector<Point2f> points2D;
    vector<Point3f> points3D;

    points3D.push_back(Point3f(0, 0, 0));
    points3D.push_back(Point3f(size, 0, 0));
    points3D.push_back(Point3f(0, size, 0));
    points3D.push_back(Point3f(0, 0, -size));

    projectPoints(points3D, rvec, tvec, cameraMatrix, distCoef, points2D);

    line(image, points2D[0], points2D[1], Scalar(0, 0, 255), 3);
    line(image, points2D[0], points2D[2], Scalar(0, 255, 0), 3);
    line(image, points2D[0], points2D[3], Scalar(255, 0, 0), 3);
}

bool readIntrinsicParams(String intrinsicsFile, Mat &camera, Mat &distortion)
{
    FileStorage file(intrinsicsFile, 0);
    if (!file.isOpened())
    {
        return false;
    }
    file["camera_matrix"] >> camera;
    if (camera.empty())
    {
        file["camera-matrix"] >> camera;
    }

    file["distortion_coefficients"] >> distortion;
    if (distortion.empty())
    {
        file["distortion-coefficients"] >> distortion;
    }

    if (camera.empty() && distortion.empty())
    {
        return false;
    }
    file.releaseAndGetString();
    return true;
}

void generate3D(int rows, int cols, int size, vector<Point3f> &corners3D)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            corners3D.push_back(Point3f(j * size, i * size, 0));
        }
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
        String inputFile = parser.get<String>(0);
        String outputFile = parser.get<String>("out");
        int rows = parser.get<int>("r");
        int cols = parser.get<int>("c");
        int size = parser.get<int>("s");
        String intrinsicsFile = parser.get<String>(1);
        String inputFile2 = parser.get<String>("i");

        // Variables needed
        Mat frame;
        Mat grayFrame;
        VideoCapture scene;
        TermCriteria termCrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        vector<Point2f> corners2D;
        vector<Point3f> corners3D;
        Mat cameraMatrix;
        Mat distCoef;
        Mat rvec;
        Mat tvec;

        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }

        generate3D(rows, cols, size, corners3D);
        scene.open(inputFile);
        if (readIntrinsicParams(intrinsicsFile, cameraMatrix, distCoef) == false)
        {
            cout << "Error. Could not read the intrinsic file" << endl;
            exit(-1);
        }

        while (true)
        {
            scene.read(frame);
            if (findChessboardCorners(frame, Size(cols, rows), corners2D, CALIB_CB_FAST_CHECK))
            {
                cvtColor(frame, grayFrame, CV_BGR2GRAY);
                cornerSubPix(grayFrame, corners2D, Size(5, 5), Size(-1, -1), termCrit);
                solvePnP(corners3D, corners2D, cameraMatrix, distCoef, rvec, tvec);

                drawCoordenateAxis(frame, size, cameraMatrix, distCoef, rvec, tvec);
            }
            namedWindow("Video", WINDOW_AUTOSIZE);
            imshow("Video", frame);
            if (waitKey(25) == 27)
            {
                break;
            }
        }
        scene.release();
        destroyWindow("Video");
    }
    catch (exception &e)
    {
        cerr << "Capturada excepcion" << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
