#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

//New libraries
#include <vector>

using namespace cv;
using namespace std;

//Structure to keep the color of a pixel
struct rgbPoint
{
  double r=0.0;
  double g=0.0;
  double b=0.0;
  double total = r + g + b;
}typedef rgbPoint;

struct sliderPararm
{
  String input;
  vector<rgbPoint> brightPixels;
  rgbPoint maxValues;
};

/*
  FUNCTION THAT GETS THE RGB VALUES OF ALL THE IMAGE'S PIXELS
*/
void getRGBPixelValue(Mat &image, vector<rgbPoint> &brightPixels, rgbPoint maxValues){
  for (int i = 0; i < image.rows; ++i)
  {
    for(int j = 0; j < image.cols; ++j){
      rgbPoint pixel;

      uchar *pointer = image.ptr<uchar>(i,j);

      pixel.b = (double) pointer[2];
      pixel.g = (double) pointer[1];
      pixel.r = (double) pointer[1];

      brightPixels.push_back(pixel);

      if(pixel.total > maxValues.total){
        maxValues=pixel;
      }
    }
  }
}

/*
  FUNTION THAT SORTS THE RGB VECTOR AND KEEPS IN THE FIRST POSITION THE BRIGHTEST ONE
*/
bool sortVector(rgbPoint small, rgbPoint big){
  return (small.total < big.total);
}

/*
  FUNCTION THAT CALIBRATES THE IMAGE
*/
void calibrateColor(Mat &image, rgbPoint &maxValues){
  for (int i = 0; i < image.rows; ++i)
  {
    for(int j = 0; j < image.cols; ++j){
      uchar *pointer = image.ptr<uchar>(i,j);

      if(pointer[2] >= maxValues.b){
        pointer[2]=255;
      }
      if(pointer[2] != 255){
        pointer[2]=((double)pointer[2]*255)/maxValues.b;
      }

      if(pointer[1] >= maxValues.g){
        pointer[1]=255;
      }
      if(pointer[1] != 255){
        pointer[1]=((double)pointer[1]*255)/maxValues.g;
      }

      if(pointer[0] >= maxValues.r){
        pointer[0]=255;
      }
      if(pointer[0] != 255){
        pointer[0]=((double)pointer[0]*255)/maxValues.r;
      }
    }
  }
  imshow("Restulted image", image);
}

/*
  FUNCTION THAT CALCULATES THE MEANS OF THE P NEIGHBOUR FOR CALIBRATION
*/
void calculateMeanRGB(Mat image, double p, vector <rgbPoint> brightPixels, rgbPoint &maxValues){
  double numPixel = (brightPixels.size()*p);
  double posPixel = brightPixels.size()-1;
  double actualPixel = 0;

  if(p > 1 || p<0){
    cout<<"Error! The percentage must not be greater than 1"<<endl;
    exit(-1);
  }
  if (p<=1 && p>0){
    while(actualPixel<numPixel){
      if(posPixel == brightPixels.size()-1){
        maxValues.b = brightPixels[posPixel].b;
        maxValues.g = brightPixels[posPixel].g;
        maxValues.r = brightPixels[posPixel].r;
      }
      else{
        maxValues.b = maxValues.b + brightPixels[posPixel].b;
        maxValues.g = maxValues.g + brightPixels[posPixel].g;
        maxValues.r = maxValues.r + brightPixels[posPixel].r;
      }

      actualPixel++;
      posPixel--;
    }
    //Now we get the portion that has to increase each color
    maxValues.b = maxValues.b/numPixel;
    maxValues.g = maxValues.g/numPixel;
    maxValues.r = maxValues.r/numPixel;
  }else if(p == 0){
    maxValues.b = brightPixels[posPixel].b;
    maxValues.g = brightPixels[posPixel].g;
    maxValues.r = brightPixels[posPixel].r;
  }
  //Now we calibrate the color
  calibrateColor(image, maxValues);
}

void calculateMeanRGBSlider(int something, void* param){
  sliderPararm *paramS = static_cast<sliderPararm*>(param);
  vector<rgbPoint> brightPixels = paramS->brightPixels;
  rgbPoint maxValues = paramS->maxValues;
  String input = paramS->input;
  Mat image = imread(input);

  double p = something;
  double numPixel = (brightPixels.size())*p/100;
  double posPixel = brightPixels.size()-1;
  double actualPixel = 0;

  while(actualPixel<numPixel){
    if(posPixel == brightPixels.size()-1){
      maxValues.b = brightPixels[posPixel].b;
      maxValues.g = brightPixels[posPixel].g;
      maxValues.r = brightPixels[posPixel].r;
    }
    else{
      maxValues.b = maxValues.b + brightPixels[posPixel].b;
      maxValues.g = maxValues.g + brightPixels[posPixel].g;
      maxValues.r = maxValues.r + brightPixels[posPixel].r;
    }

    actualPixel++;
    posPixel--;
  }
  //Now we get the portion that has to increase each color
  maxValues.b = maxValues.b/numPixel;
  maxValues.g = maxValues.g/numPixel;
  maxValues.r = maxValues.r/numPixel;
  if(numPixel == 0){
    maxValues.b = brightPixels[posPixel].b;
    maxValues.g = brightPixels[posPixel].g;
    maxValues.r = brightPixels[posPixel].r;
  }
  //Now we calibrate the color
  calibrateColor(image, maxValues);
}

const String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
    "{p               |     | percentage of neighbour point}"
    "{i               |     | interactive mode      }"
    ;

int main (int argc, char* const* argv)
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

      if(parser.has("p") && parser.has("i"))
      {
        cout<<"Error! Please, do not use the option -p and -i at the same time"<<endl;
        return -1;
      }
      //New variables
      String input = parser.get<String>("path");
      float p = parser.get<float>("p");
      Mat image = imread(input);

      //New variables
      vector <rgbPoint> brightPixels;
      rgbPoint maxValues;
      int sliderValue=0;
      sliderPararm param;

      getRGBPixelValue(image, brightPixels, maxValues);

      sort(brightPixels.begin(), brightPixels.end(), sortVector);
      namedWindow("Restulted image", WINDOW_AUTOSIZE);
      if(parser.has("i")){
        namedWindow("Original image", WINDOW_AUTOSIZE);
        param.input = input;
        param.brightPixels = brightPixels;
        param.maxValues = maxValues;    
        createTrackbar("Pertcentage", "Original image", &sliderValue, 100, calculateMeanRGBSlider, &param);
        imshow("Original image", image); 
        imshow("Restulted image", image);
        waitKey(0);
      }
      else{
        calculateMeanRGB(image, p, brightPixels, maxValues);
        waitKey(0);
      }
  }
  catch (exception& e)
  {
    cerr << "Capturada excepcion: " << e.what() << endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
