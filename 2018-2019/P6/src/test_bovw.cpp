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
    "{img             |   | filename for image             }"
    "{vid             |   | filename for the video             }"
    "{dict             |.   | Dictionary file .yml}"
    "{classifier             |.   | Classifier file .yml}" 
    "{desc_t             | 0   | descriptor type used for test. Default: 0(SIFT). 0->SIFT. 1->SURF. 2->Dense SIFT}" 
    "{threshold             | 200   | threshold used for the descriptor SURF}" 
    "{steps            | 400 | distance from where you take the keypoint(Dense SIFT). Default: 400}"
    "{n_k            | 1 | Number neightbours used in knn}"
    "{n_c            | 1 | parameter C of the svm}"
    "{n_b            | 100 | Number of weak counts [Boosting]}"
    "{class_type            | knn | type of the classifier used}"
    "{svm_type            | linear | kernel used in svm}"
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
        int c_value = parser.get<int>("n_c");
        int b_value = parser.get<int>("n_b");
        int k_value = parser.get<int>("n_k");

        string classifier_type = parser.get<string>("class_type");
        string svm_kernel = parser.get<string>("svm_type");

        // New variables
        vector<string> categories;
        vector<int> samples_per_category;
        vector<Mat> images;
        vector<float> true_labels;
        vector<Mat> X_test;
        vector<Mat> BOVW;

        int keywords = 0;
        Mat test_bovw;
        Mat predicted_labels;

        X_test.clear();

        FileStorage dictFile;
        FileStorage classFile;

        dictFile.open(dictName, FileStorage::READ);
        classFile.open(className, FileStorage::READ);

        dictFile["keywords"]>>keywords;

        Ptr<ml::KNearest> dictionary = Algorithm::read<ml::KNearest>(dictFile.root());
        dictFile.release();
        Ptr<ml::StatModel> classifier;
        
        if(classifier_type == "knn"){
            clog<<"The user has choosed KNN classifier"<<endl;
            FileNode fn = classFile["opencv_ml_knn"];
            for(FileNodeIterator it = fn.begin(); it != fn.end(); it++){
                FileNode item = *it;
                if(item.name() == "default_k"){
                    k_value = (int)item;
                }
            }
            Ptr<ml::KNearest> knnClasiffier = Algorithm::load<ml::KNearest>(className);
            knnClasiffier->setDefaultK(k_value);
            classifier = knnClasiffier;
        }else if(classifier_type == "svm"){

            clog<<"The user has choosed SVM classifier"<<endl;
            FileNode fn = classFile["opencv_ml_svm"];
            for(FileNodeIterator it = fn.begin(); it != fn.end(); it++){
                FileNode item = *it;
                if(item.name() == "C"){
                    c_value = (float)item;
                }
            }
            Ptr<ml::SVM> svmClassifier = Algorithm::load<ml::SVM>(className);
            if(svm_kernel == "linear"){
                    svmClassifier->setKernel(ml::SVM::LINEAR);
                }else if(svm_kernel == "polynomial"){
                    svmClassifier->setKernel(ml::SVM::POLY);
                    svmClassifier->setDegree(1);
                }else if(svm_kernel == "radial"){
                    svmClassifier->setKernel(ml::SVM::RBF);
                }else{
                    cerr<<"Not kernel identified... Exiting program..."<<endl;
                    exit(-1);
                }
                svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
                svmClassifier->setType(ml::SVM::C_SVC);
                svmClassifier->setC(c_value);
                classifier = svmClassifier;
        }else if(classifier_type == "boosting"){
            clog<<"The user has choosed Boosting classifier"<<endl;
            FileNode fn = classFile["opencv_ml_boost"];
            for(FileNodeIterator it = fn.begin(); it != fn.end(); it++){
                FileNode item = *it;
                if(item.name() == "ntrees"){
                    b_value = (float)item;
                }
            }
            Ptr<ml::Boost> boostingClassifier = Algorithm::load<ml::Boost>(className);
            boostingClassifier->setBoostType(ml::Boost::DISCRETE);
            boostingClassifier->setWeakCount(b_value);
            classifier = boostingClassifier;
        }
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

        // If it is an image
        if(parser.has("img")){

            Mat image =  imread(fileName, 0);
            Mat descs;

            if(image.empty()){
                cerr<<"Couldn't read the image...Exiting program..."<<endl;
                exit(-1);
            }else{
                resize(image, image, Size(IMG_WIDTH, round(IMG_WIDTH * image.rows/image.cols)));
                if(desc_t == 0){
                    
                    descs = extractSIFTDescriptors(image, keywords);
                    
                }else if(desc_t == 1){
                    descs = extractSURFdescriptors(image, threshold);
                }else{
                    descs = extractDenseSIFTdescriptors(image, keywords, steps);
                }
            }

            namedWindow("Original image", WINDOW_AUTOSIZE);
            imshow("Original image", image );
            cout<<keywords<<endl;
            clog<<"\tClassifing test images..."<<endl;

            vector<Mat> BOVW;
            
            BOVW.push_back(compute_bovw(dictionary, keywords, descs));

            vconcat(BOVW, test_bovw);
 
            classifier->predict(test_bovw, predicted_labels);
  
            imshow("Predicted category->"+categories[predicted_labels.at<float>(0)], image);
            
            clog<<"Predicted category->"<<categories[predicted_labels.at<float>(0)]<<endl;
            waitKey(0);
        }else{
            // If it is a video
            Mat frame;
            VideoCapture scene(parser.get<string>("vid"));
            cout<<parser.get<string>("vid")<<endl;
            while(scene.read(frame)){
                Mat image;
                cvtColor(frame, image, COLOR_RGB2GRAY);
                resize(image, image, Size(IMG_WIDTH, round(IMG_WIDTH * image.rows/image.cols)));
                Mat descs;
                if(desc_t == 0){
                    descs = extractSIFTDescriptors(image, keywords);
                }else if(desc_t == 1){
                    descs = extractSURFdescriptors(image, threshold);
                }else{
                    descs = extractDenseSIFTdescriptors(image, keywords, steps);
                }

                Mat prediction;
                
                test_bovw = compute_bovw(dictionary, keywords, descs);
                classifier->predict(test_bovw, prediction);
                
                imshow("Predicted category->"+categories[prediction.at<float>(0)], frame);
                
                if (waitKey(5) >= 0){
                    break;
                }
            }
        }
    }
    catch (exception& e)
    {
        cerr << "Capturada excepcion: " << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
