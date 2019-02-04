#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp> 
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp> 
#include <iomanip>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>


#include "common_code.hpp"

#define IMG_WIDTH 300

using namespace std;
using namespace cv;

const String keys =
    "{help h usage ? |      | print this message   }"
    "{config_file              |     | configuration file for the dataset      }"
    "{img             | ,  | filename for image             }"
    "{dict             |.   | Dictionary file .yml}"
    "{classifier             |.   | Classifier file .yml}" 
    "{desc_t             | 0   | descriptor type used for test. Default: 0(SIFT). 0->SIFT. 1->SURF. 2->Dense SIFT}" 
    "{threshold             | 200   | threshold used for the descriptor SURF}" 
    "{steps            | 10 | distance from where you take the keypoint(Dense SIFT). Default: 400}"
    ;

int main(int argc, char const *argv[])
{
    int retCode = EXIT_SUCCESS;
    try{
        CommandLineParser parser(argc, argv, keys);
        parser.about("Application name v1.0.0");
        if(parser.has("help")){
          parser.printMessage();
          return 0;
        }

        if(!parser.check()){
            parser.printErrors();
            return 0;
        }
        // Command variables
        string config_path = parser.get<string>("config_file");
        string fileName = parser.get<string>("img");
        string dictName = parser.get<string>("dict");
        string className = parser.get<string>("classifier");
        int desc_t = parser.get<int>("desc_t");
        string image_path = parser.get<string>("img");
        int threshold = parser.get<int>("threshold");
        int steps = parser.get<int>("steps");

        // New variables
        vector<string> categories;
        vector<int> samples_per_category;
        vector<Mat> images;
        vector<float> true_labels;
        vector<Mat> X_test;
        vector<Mat> BOVW;

        int keywords,default_k;
        Mat img;
        Mat test_bovw;
        Mat predicted_labels;

        X_test.clear();

        cv::FileStorage dictFile;
        cv::FileStorage classFile;

        dictFile.open(dictName, cv::FileStorage::READ);
        classFile.open(className, cv::FileStorage::READ);

        dictFile["keywords"]>>keywords;

        cv::Ptr<cv::ml::KNearest> dictionary = cv::Algorithm::read<cv::ml::KNearest>(dictFile.root());
        dictFile.release();
        cv::Ptr<cv::ml::KNearest> classifier = cv::Algorithm::load<cv::ml::KNearest>(className);
        classFile.release();

        int retCode;
        if((retCode = load_dataset_information(config_path, categories, samples_per_category)) != 0){
            cerr << "Error: could not load dataset information from '"<< config_path << "' (" << retCode << ")." << endl;
            exit(-1);
        }

        cout<<"Found "<<categories.size()<<" categories"<<endl;
        if(categories.size() < 2){
            cerr<<"Error: at least two categories are needed."<<endl;
            exit(-1);
        }

        cout<<"The categories are: "<<endl;
        for(int i = 0; i < categories.size(); i++){
            cout<<"\t"<<categories[i]<<endl;
        }
        clog<<"Testing..."<<endl;
        clog<<"\tComputing image descriptor for test image"<<endl;
        img = imread(image_path, IMREAD_GRAYSCALE);
        if(img.empty()){
            cerr<<"Error: The image could not be read"<<endl;
            exit(-1);
        }
        images.push_back(img);
        
        for(size_t c = 0; c < images.size(); c++){
            // Image resize
            true_labels.push_back(c);
            Size descriptorSize = Size(IMG_WIDTH, images[c].rows);
            resize(images[c], images[c], descriptorSize);

            // Extract descriptors
            Mat descs;
            
            
            if(desc_t == 0){
                descs = extractSIFTDescriptors(images[c], keywords);
            }else if(desc_t == 1){
                descs = extractSURFdescriptors(images[c], threshold);
            }else{
                descs = extractDenseSIFTdescriptors(images[c], keywords, steps);
            }
            X_test.push_back(descs);
        }

        namedWindow("Original image", WINDOW_AUTOSIZE);
        imshow("Original image", img );

        BOVW.push_back(compute_bovw(dictionary, keywords, X_test[0]));
        vconcat(BOVW, test_bovw);

        clog<<"There are "<<test_bovw.size()<<" images for the test"<<endl;
        clog<<"\tClassifying..."<<endl;
        
        cout<<"How many neighbour do you want? ";
        cin>>default_k;
        if(default_k <= 0){
            cerr<<"Error: The number of neighbours should be greater than 0"<<endl;
            exit(-1);
        }

        classifier->findNearest(test_bovw, default_k, predicted_labels);

        Mat confussion_matrix = compute_confusion_matrix(categories.size(), Mat(true_labels), predicted_labels);
        CV_Assert(int(sum(confussion_matrix)[0]) == test_bovw.rows);

        imshow("Predicted category->"+categories[predicted_labels.at<float>(0)], img);
        clog<<"Predicted category->"<<categories[predicted_labels.at<float>(0)]<<endl;
        waitKey(0);
    }
    catch (exception& e)
    {
        cerr << "Capturada excepcion: " << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
