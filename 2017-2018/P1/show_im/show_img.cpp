#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

const String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
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
      String img1 = parser.get<String>("path");
      //Control for the argument line
      if(argc<2){
        cout<<"Error! Please, try:"<<endl;
        cout<<"./show_img <img>"<<endl;
      }
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

  	//Create a matrix to keep the image
  	Mat image;
  	image = imread(img1, 1);

  	if(!image.data){
          printf("No image data\n");
          return -1;
      }

      namedWindow("Show image", WINDOW_AUTOSIZE);
      imshow("Show image", image);

      waitKey(0);
    
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
