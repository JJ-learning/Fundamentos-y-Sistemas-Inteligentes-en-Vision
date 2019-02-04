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

struct sliderParam{
  String input;
  float percentile;
  float sigma;
  float percentile2;
  int mode;
}typedef sliderParam;

/*
  FUNCTION THAT APPLIES THE SOBEL DETECTOR TO AN IMAGE
*/
void applySobelFilter(Mat image, float percentile){
  Mat gradientX, gradientY, thresholdized;

  namedWindow("Original image", WINDOW_AUTOSIZE);
  imshow("Original image", image);

  Sobel(image, gradientX, CV_8UC1, 1, 0, 3);
  Sobel(image, gradientY, CV_8UC1, 0, 1, 3);

  normalize(gradientX, gradientX, 0, 255, NORM_MINMAX);
  normalize(gradientY, gradientY, 0, 255, NORM_MINMAX);

  namedWindow("Gradient X", WINDOW_AUTOSIZE);
  imshow("Gradient X", gradientX);

  namedWindow("Gradient Y", WINDOW_AUTOSIZE);
  imshow("Gradient Y", gradientY);

  threshold(image, thresholdized, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  namedWindow("thresholdized image", WINDOW_AUTOSIZE);
  imshow("thresholdized image", thresholdized);

  waitKey(0);
}

/*
  FUNCTION THAT APPLIES THE LAPLACIAN FILTER IN ORDER TO DETECT BORDERS
*/
void applyLaplacianFilter(Mat inputImage, float sigma, float percentile){
  int kernelSize=(sigma*2)+1;
  Mat laplace;
  //If sigma is 0 then we don't smooth the image
  if(kernelSize>0 && (kernelSize%2)!=0){
    GaussianBlur(inputImage, inputImage, Size_<int>(kernelSize, kernelSize), 0, 0, BORDER_DEFAULT);
  }
  Laplacian(inputImage, laplace, CV_32FC1);
  laplace = laplace > percentile;
  imshow("Laplace filter", laplace);
}

/*
  FUNCTION THAT APPLIES THE CANNY DETECTOR TO AN IMAGE
*/
void applyCannyFilter(Mat inputImage, float sigma, float highThres, float lowThres){
  int kernelSize = (sigma*2)+1;
  Mat gradientX, gradientY, output;

  if(kernelSize>0 && (kernelSize%2)!=0){
    GaussianBlur(inputImage, inputImage, Size_<int>(kernelSize, kernelSize), sigma, sigma, BORDER_DEFAULT);
  }

  Sobel(inputImage, gradientX, CV_16SC1,1,0,3);
  Sobel(inputImage, gradientY,CV_16SC1, 0,1,3);

  Canny(gradientX, gradientY, output, highThres, lowThres, false);

  imshow("Canny detector", output);
}

/*
  FUNCTION THAT APPLIES THE SLIDER TO AN IMAGE
*/
void percentileTrackbar(int percentile, void* param){
  sliderParam *paramS = static_cast<sliderParam*>(param);
  Mat image = imread(paramS->input, 0);
  paramS->percentile = (float)percentile/10;
  if(paramS->mode == 1){
    applyLaplacianFilter(image, paramS->sigma, paramS->percentile);
  }else if(paramS->mode == 3){
    applyCannyFilter(image, paramS->sigma, paramS->percentile, paramS->percentile2);
  }  
}

/*
  FUNCTION THAT APPLIES THE SLIDER TO AN IMAGE
*/
void sigmaTrackbar(int sigma, void* param){
  sliderParam *paramS = static_cast<sliderParam*>(param);
  Mat image = imread(paramS->input, 0);
  paramS->sigma = (float) sigma;
  if(paramS->mode == 1){
    applyLaplacianFilter(image, paramS->sigma, paramS->percentile); 
  }else if(paramS->mode == 3){
    applyCannyFilter(image, paramS->sigma, paramS->percentile, paramS->percentile2);
  }   
}

/*
  FUNCTION THAT APPLIES THE SLIDER TO AN IMAGE
*/
void percentile2Trackbar(int percentile, void* param){
  sliderParam *paramS = static_cast<sliderParam*>(param);
  Mat image = imread(paramS->input, 0);
  
  paramS->percentile2 = percentile/10;
  applyCannyFilter(image, paramS->sigma, paramS->percentile, paramS->percentile2);
}


const String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
    "{t               | 0   | type of detector      }"
    "{p               | 0.9 | high threshold        }"
    "{s               | 0.5 | Gaussian sigma        }"
    "{P               | 0.1 | low threshold         }"
    "{i               |     | interactive mode      }"
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
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

    /*Ahora toca que tu rellenes con lo que hay que hacer ...*/

      String path = parser.get<String>("path");

      //Required variables for sobel function
      int typeDetector=parser.get<int>("t");
      float percentile = parser.get<float>("p");
      Mat inputImage = imread(path, 0);

      //Required variables for Laplacian function
      sliderParam param;
      float sigma;

      //Required variables for Canny detector
      float percentile2;

      switch(typeDetector){
        case 0: //TO CONTINUE
            applySobelFilter(inputImage, percentile);
          break;
        case 1:
            namedWindow("Laplace filter", WINDOW_AUTOSIZE);
            sigma = parser.get<float>("s");

            if(parser.has("i")){
              sigma=0;
              percentile=0;
              param.input = path;
              param.percentile = percentile;
              param.sigma = sigma;
              param.mode = typeDetector;

              imshow("Laplace filter", inputImage);
              createTrackbar("Sigma", "Laplace filter", (int*)&param.sigma, 31, sigmaTrackbar, &param);
              createTrackbar("Percentile (%)", "Laplace filter", (int*)&param.percentile, 100, percentileTrackbar, &param);
            }else{
              applyLaplacianFilter(inputImage, sigma, percentile);              
            }
            waitKey(0);
          break;
        case 3:
            namedWindow("Canny detector", WINDOW_AUTOSIZE);
            sigma = parser.get<float>("s");
            percentile2 = parser.get<float>("P");
            if(parser.has("i")){
              sigma=0;
              percentile=0;
              percentile2=0;
              param.input = path;
              param.percentile = percentile;
              param.percentile2 = percentile2;
              param.sigma = sigma;
              param.mode = typeDetector;

              imshow("Canny detector", inputImage);
              createTrackbar("Sigma", "Canny detector", (int*)&param.sigma, 31, sigmaTrackbar, &param);
              createTrackbar("Low threshold (%)", "Canny detector", (int*)&param.percentile, 100, percentileTrackbar, &param);
              createTrackbar("High threshold (%)", "Canny detector", (int*)&param.percentile2, 100, percentile2Trackbar, &param);

            }else{
              applyCannyFilter(inputImage, sigma, percentile, percentile2);
            }
            waitKey(0);
          break;
        default:
          break;
      }
    
  }
  catch (exception& e)
  {
    cerr << "Capturada excepcion: " << e.what() << endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
