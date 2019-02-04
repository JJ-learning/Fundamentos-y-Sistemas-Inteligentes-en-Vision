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


#include "common.hpp"

#define IMG_WIDTH 300

using namespace std;
using namespace cv;

const String keys =
    "{help h usage ? |      | print this message   }"
    "{b              |     | basename for the dataset      }"
    "{c              |     | configuration file for the dataset      }"
    "{f              | ,  | filename for image             }"
    "{ntrain            | 15 | Number of samples per class used to train }"
    "{ntest            | 50 | number of samples per class used to test}"
    "{NN               | 1 | [Kmeans] Number of neihgors used to classify}"
    "{mode            | 1 | Classifier mode}"
    "{dict             |.   | Dictionary file .yml}"
    "{class             |.   | Dictionary file .yml}"   
    ;


int
main(int argc, char* const* argv){
    
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
        //get the command variables
        int ntrain = parser.get<int>("ntrain");
        int ntest = parser.get<int>("ntest");
        int NNnumber = parser.get<int>("NN");
        int classMode = parser.get<int>("mode");
        string basename = parser.get<string>("b");
        string config_path = parser.get<string>("c");
        string fileName = parser.get<string>("f");
        string dictName = parser.get<string>("dict");
        string className = parser.get<string>("class");
        string dataset_desc_file = basename + '/' + config_path;
        fileName = basename + '/' + fileName;

        Ptr<ml::KNearest> KNN;
        Ptr<ml::KNearest> KNNResult;
        Ptr<ml::RTrees> RFTDictionary;
        Ptr<ml::RTrees> RFTResult;

        if(!parser.has("dict") && !parser.has("class")){
            dictName = "Dictionary.yml";
            className = "Classifier.yml";
        }  

        if(classMode==1){
            KNN = KNN->load<ml::KNearest>(dictName);
            KNNResult = KNNResult-> load<ml::KNearest>(className);
        }else if(classMode==2){
            RFTDictionary = RFTDictionary->load(dictName);
            RFTResult = RFTResult->load(className);
        }
        
        

        Mat image, img;
        Mat test_bovw;
        Mat predict_labels;
        Mat trainData;
        Mat confusion_mat;

        vector<string> categories;
        vector<int> samples_per_cat;
        vector<float> true_labels;
        vector<Mat> images;
        vector<Mat> imagesTest; 
        vector<Mat> X_test;
        vector<Mat> BoVW;
        vector<vector<int>> train_samples;
        vector<vector<int>> test_samples;

        double rRate_dev=0, rRate_mean=0;

        X_test.clear();
        BoVW.clear();
        
        int retCode;
        if ((retCode = load_dataset_information(dataset_desc_file, categories, samples_per_cat)) != 0)
        {
            cerr << "Error: could not load dataset information from '"
                << dataset_desc_file
                << "' (" << retCode << ")." << endl;
            exit(-1);
        }

        cout << "Found " << categories.size() << " categories: ";
        if (categories.size()<2)
        {
            cerr << "Error: at least two categories are needed." << endl;
            return -1;
        }

        for (size_t i=0;i<categories.size();++i){
            cout << categories[i] << ' ';
        }
        cout << endl;

       clog<<"Testing ..."<<endl;
       clog<<"\tCompute image descriptor for test image..."<<endl;
        image=imread(fileName, IMREAD_GRAYSCALE);
        images.push_back(image);
       for (size_t c = 0; c < images.size(); ++c)
        {
                // Image resize
                //-----------------------------------------------------
                true_labels.push_back(c);
                Size descripsize=Size(IMG_WIDTH, image.rows);
                //cout<<"File name: "<<fileName<<endl;

                namedWindow("Original Image", WINDOW_AUTOSIZE);
                imshow("Original Image", image);
                
                if(image.empty()){
                    clog<<"Error: empty image... Exit."<<endl;
                    exit(-1);
                }
                resize(image, image, descripsize);
                // Extract SIFT descriptors
                //-----------------------------------------------------
                Mat descs;
                descs = extractSIFTDescriptors(image, 0);
                X_test.push_back(descs); 
        }
        waitKey(0);
        switch(classMode){
            case 1:
                 BoVW.push_back(compute_bovw(KNN, 100, X_test[0], true));
            break;
            case 2:
                BoVW.push_back(compute_bovw(RFTDictionary, 100, X_test[0], true));
            break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
       
       
       vconcat(BoVW, test_bovw);

       clog<<"\tThere are "<<test_bovw.rows<<" test images"<<endl;
       clog<<"\tClassifing test images ..."<<endl;
       switch(classMode){
            case 1:
                 KNNResult->findNearest(test_bovw, NNnumber, predict_labels);
            break;
            case 2:
                RFTResult->predict(test_bovw,predict_labels);
            break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
       confusion_mat = compute_confusion_matrix(categories.size(), Mat(true_labels), predict_labels);
        CV_Assert(int(sum(confusion_mat)[0]) == test_bovw.rows);
        imshow("Confusion Matrix", confusion_mat);

       imshow("Predicted category->"+categories[predict_labels.at<float>(0)], image);
       cout<<"Predicted category->"+categories[predict_labels.at<float>(0)]<<endl;
       waitKey(0);

    }
    catch (exception& e)
    {
        cerr << "Capturada excepcion: " << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
