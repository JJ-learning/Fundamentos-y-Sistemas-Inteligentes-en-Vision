#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

//Namespaces 
using namespace cv;
using namespace std;

const String keys =
    "{help h usage ? |      | print this message   }"
    "{path        | .     | path to the file   }"
    "{t              | 60  | number of ms to wait  }"
    ;
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
      int Time = parser.get<int>("t");//Number of ms
      String path = parser.get<String>("path");
      VideoCapture capture;
    //Control for the argument line
      if (argc<2)
      {
        cout<<"Error! Please try:"<<endl;
        cout<<"./show_video <video>"<<endl;
      }

      if(path=="0" || path=="-1"){
        capture.open(0); //Open camera
      }
      else
        capture.open(path);
  	  
      if(!capture.isOpened()){
          printf("No video data\n");
          return -1;
      }
      namedWindow("Show video", WINDOW_AUTOSIZE);
      for(;;){
          Mat frame;
          
          capture.read(frame);//We get a new frame from camera
          if(frame.empty()){
              printf("ERROR! Blank frame stored\n");
              break;
          }        
          //Show the video and close it if the user press "ESC"
          imshow("Show video", frame);
          //It includes Time to say how much time it has to wait until the next frame
          if(waitKey(Time)==27)
          {
              capture.release();
              destroyWindow("Show video");
              break;//'ESC' key pressed, finish the program
          }
                 
      }
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
