#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//New libraries
#include <vector>

//Namespaces 
using namespace cv;
using namespace std;

const String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
    "{out            |      | intrinsic file       }"
    "{r              |  5   | number of rows        }"
    "{c              |  6    | number of colunms        }"
    "{s              |  3    | size                  }"
    ;
//New key has been added in order to control the number of ms between frames

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      CommandLineParser parser(argc, argv, keys);
      parser.about("Application name v1.0.0");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      String video = parser.get<String>("path");
      String intrinsicFile = parser.get<String>("out");
      //Required arguments
      int rows = parser.get<int>("r");
      int cols = parser.get<int>("c");
      int size = parser.get<int>("s");

      //New variables
      int stop, stop2;
      Mat frame; //Video's frame
      Mat grayFram; //Gray video's frame
      vector<vector<Point3f>> objectPoints; //For the physical position of the corners 3D points
      vector<vector<Point2f>> imagePoints; //For the location of the corners in the image 2D points
      vector<Point2f> corners; //Keeps the 2D points of the chess
      vector <Point3f> obj; //Keeps thevertex's positions
      Size boardSize = Size(rows-1, cols-1);
      int maxFrames = 0;//To see if there are more frames than the value size
      
      //Variables to calibrate the camera
      Mat intrinsic;
      Mat distortionCoef;
      vector<Mat> rotate;
      vector<Mat> translate;

      VideoCapture capture(video); //Open camera
      if(!capture.isOpened()){
         printf("No video data\n");
         return -1;
      }
      
      /*Keeps the position of each vertex*/
      for (int i = 0; i < rows-1; ++i){
         for(int j=0;j<cols-1;j++){
            obj.push_back(Point3f(i*size, j*size, 0));
         }
      }

      for(;;){
         capture.read(frame);//We get a new frame from camera
         if(frame.empty()){
            printf("ERROR! Blank frame stored\n");
            break;
         }
              
         namedWindow("Show video", WINDOW_AUTOSIZE);
         imshow("Show video", frame);
         stop = waitKey();
         //The user press 'f'
         if(stop==102 || stop==32){
          //Next frame 
         }
         //The user press 'p'
         else if(stop==112){
            cvtColor(frame, grayFram, COLOR_BGR2GRAY);
            bool found = findChessboardCorners(frame, boardSize, corners);
            //If corners detected
            if(found)
            {
               cornerSubPix(grayFram, corners, Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
               drawChessboardCorners(grayFram, boardSize, corners, found);
            }
            //We show the image
            imshow("Gray", grayFram);
            stop2=waitKey();
            //If the user agrees and press SPACE
            if (stop2==32)
            {
               imagePoints.push_back(corners);
               objectPoints.push_back(obj);
               cout<<"frame stored successfully"<<endl;
               maxFrames = maxFrames +1;
               //We have reached the maximun number of keeped frames
               if(maxFrames>=size){
                  stop=27;
                  break;
               }
            }
         }
         else if(stop==27){
            break;
         }
         else
            break;            
      }

      if(stop==27){
         //Calibrate the camera         
         double error=calibrateCamera(objectPoints, imagePoints, frame.size(), intrinsic, distortionCoef, rotate, translate);
         FileStorage finalFile(intrinsicFile, FileStorage::WRITE);

         finalFile<<"image-width"<<frame.size().width;
         finalFile<<"image-height"<<frame.size().height;
         finalFile<<"error"<<error;
         finalFile<<"camera-matrix"<<intrinsic<<"distortion-coefficients"<<distortionCoef;
         finalFile<<"rotation-matrix"<<rotate;
         finalFile<<"translation-matrix"<<translate;
         cout<<"File saved successfully"<<endl;
         finalFile.release();
         capture.release();
      }      
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
