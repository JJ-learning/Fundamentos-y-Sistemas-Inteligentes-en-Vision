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


//Required variables
bool hsvKey=false;
int filterSize;
int filterMode;
double gain;
Mat dest; //Resulted matrix
Mat input;
Point point=Point(-1,-1); //Anchor point
Mat output;
Mat mask;
Mat channels[3]; //Final matrix with the splitted inputmatrix

static void on_trackbarG(int something,void* ){
    gain = something;
    if(hsvKey == false){
        //Choose the used filter
        switch(filterMode){
            case 0://Box filter
                boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                output = (gain+1)*input - gain*dest;
                break;
            case 1:
                GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                output = (gain+1)*input - gain*dest;
                break;
            case 2:
                medianBlur(input, dest, filterSize);
                output = (gain+1)*input - gain*dest;
                break;  
            default:
                printf("Option unrecognized. Exit program...\n");
                exit(-1);
        }
    }
    else{
        Mat aux3;
        //Choose the used filter
        switch(filterMode){
            case 0://Box filter
                boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);               
                break;
            case 1:
                GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);                
                break;
            case 2:
                medianBlur(input, dest, filterSize);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);                
                break;  
            default:
                printf("Option unrecognized. Exit program...\n");
                exit(-1);
        }
    }
    imshow("Output", output);
};

static void on_trackbarF(int something,void* ){
    filterSize = something;
    if((filterSize%2)==0)
        filterSize = filterSize+1;
    if(hsvKey == false){
        //Choose the used filter
        switch(filterMode){
            case 0://Box filter
                boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                output = (gain+1)*input - gain*dest;
                break;
            case 1:
                GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                output = (gain+1)*input - gain*dest;
                break;
            case 2:
                medianBlur(input, dest, filterSize);
                output = (gain+1)*input - gain*dest;
                break;  
            default:
                printf("Option unrecognized. Exit program...\n");
                exit(-1);
        }
    }
    else{
        Mat aux3;
        //Choose the used filter
        switch(filterMode){
            case 0://Box filter
                boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);               
                break;
            case 1:
                GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);                
                break;
            case 2:
                medianBlur(input, dest, filterSize);
                output = (gain+1)*input - gain*dest;
                channels[2]=dest;
                merge(channels, 3, output);
                //Change again the color
                cvtColor(output, output, COLOR_HSV2BGR);                
                break;  
            default:
                printf("Option unrecognized. Exit program...\n");
                exit(-1);
        }
    }
    imshow("Output", output);
};

void applymask(Mat &input, Mat mask){
    Mat aux;
    input.copyTo(aux, mask);
    input = aux;
    input.rows = aux.rows;
    input.cols = aux.cols;
}

const String keys =
    "{help h usage ? |      | print this message   }"
    "{path           |.     | path to file         }"
    "{out            |.     | name output image}"
    "{r              |1   | filter size           }"
    "{f              |0     | filter mode used      }"
    "{g              |1   | gain of the filter used}"
    "{m              |<none> | mask used           }"
    "{v              |      | HSV color space       }"
    "{i              |      | interactive mode}"
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

    //Mandatory variables
    String img1 = parser.get<String>("path");
    input = imread(img1); //Input matix
    if(!input.data){
          printf("No image data\n");
          return -1;
      }
    filterSize = parser.get<int>("r");
    filterMode = parser.get<int>("f");
    gain = parser.get<double>("g");
    String outName = parser.get<String>("out");

    //Error control for variables
    if(gain<0.0 || gain>5.0){
        cout<<"Error! The gain for the image should be between 0.0 and 5.0"<<endl;
        return -1;
    }

    int maskActivated=0;
    

    //Create the window for the input image
    namedWindow("Input");
    imshow("Input", input);

    //Obtain the filter size
    filterSize = 2*filterSize+1;
    if(filterSize > (min(input.cols,input.rows)/2)){
        printf("The size of the filter is greater than allowed, closing program...\n");
        return -1;
    }
    //HSV choosed 
    if(parser.has("v")){
        hsvKey = true;
        //First, we've changed the color into HSV
        cvtColor(input, input, COLOR_BGR2HSV);
        //We split the image in order to merge later on.
        split(input, channels);
        input=channels[2];
    }
    //If a mask is used
    if (parser.has("m"))
    {
        String inputMask = parser.get<String>("m");
        mask = imread(inputMask, 0);
        applymask(input, mask);
    }

    //Show the enhanced image
    namedWindow("Output");
    //Iteractive mode choosed
    if(parser.has("i")){
        int aux2 = gain;
        //Variables: "slideName, windowApplied "
        createTrackbar("Gain Bar", "Input", &aux2, 5.0, on_trackbarG);
        createTrackbar("Filter Size", "Input", &filterSize,(min(input.rows, input.cols)/2), on_trackbarF);
        imshow("Output", input);
    }
    else{
        if(hsvKey == false){
            //Choose the used filter
            switch(filterMode){
                case 0://Box filter
                    boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                    output = (gain+1)*input - gain*dest;
                    break;
                case 1:
                    GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                    output = (gain+1)*input - gain*dest;
                    break;
                case 2:
                    medianBlur(input, dest, filterSize);
                    output = (gain+1)*input - gain*dest;
                    break;  
                default:
                    printf("Option unrecognized. Exit program...\n");
                    return -1;
            }
        }
        else{
            Mat aux3;
            //Choose the used filter
            switch(filterMode){
                case 0://Box filter
                    boxFilter(input, dest, input.depth(), Size_<int> (1,filterSize), point, false);
                    output = (gain+1)*input - gain*dest;
                    channels[2]=dest;
                    merge(channels, 3, output);
                    //Change again the color
                    cvtColor(output, output, COLOR_HSV2BGR);               
                    break;
                case 1:
                    GaussianBlur(input, dest, Size_<int> (1, filterSize), 0,0,BORDER_DEFAULT);
                    output = (gain+1)*input - gain*dest;
                    channels[2]=dest;
                    merge(channels, 3, output);
                    //Change again the color
                    cvtColor(output, output, COLOR_HSV2BGR);                
                    break;
                case 2:
                    medianBlur(input, dest, filterSize);
                    output = (gain+1)*input - gain*dest;
                    channels[2]=dest;
                    merge(channels, 3, output);
                    //Change again the color
                    cvtColor(output, output, COLOR_HSV2BGR);                
                    break;  
                default:
                    printf("Option unrecognized. Exit program...\n");
                    return -1;
            }
        }
        imshow("Output", output);
        imwrite(outName, output);
    }  
    
    waitKey(0);
    
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
