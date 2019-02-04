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
#include <math.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


#include "common.hpp"

#define IMG_WIDTH 300

using namespace std;
using namespace cv;

const string keys =
    "{help h usage ? |      | print this message   }"
    "{b              |     | basename for the dataset      }"
    "{c              |     | configuration file for the dataset      }"
    "{r               | 1 | Number of trials train/test to comute the reconition rate       }"
    "{w               | 100 | [Kmeans] Number of keywords generated        }"
    "{ntrain            | 15 | Number of samples per class used to train }"
    "{ntest            | 50 | number of samples per class used to test}"
    "{ndesc            | 0 | [SIFT] number of descriptor per image. Value 0 means extract all }"
    "{NN               | 1 | [Kmeans] Number of neihgors used to classify}"
    "{mode            | 1 | Classifier mode}"
    "{dict             |.   | Dictionary file .yml}"
    "{class             |.   | Dictionary file .yml}" 
    ;

int
main(int argc, char* const* argv){
    
    int retCode = EXIT_SUCCESS;

    //New variables
    vector<string> categories; // Keeps the name of the category
    vector<int> samples_per_cat; //Number of images that keeps the category 

    try{
        CommandLineParser parser(argc, argv, keys);
        parser.about("Application name v1.0.0");
        if(parser.has("help")){
          parser.printMessage();
          return 0;
        }


    /*                  MAIN BLOCK          */
        //get the command variables
        string basename = parser.get<string>("b");
        string config_path = parser.get<string>("c");
        float r_try = parser.get<float>("r");
        int keywords = parser.get<int>("w");
        int ntrain = parser.get<int>("ntrain");
        int ntest = parser.get<int>("ntest");
        int ndesc = parser.get<int>("ndesc");
        int NNnumber = parser.get<int>("NN");
        int classMode = parser.get<int>("mode");
        string dictName = parser.get<string>("dict");
        string className = parser.get<string>("class");
        if(!parser.has("dict") && !parser.has("class")){
            dictName = "Dictionary.yml";
            className = "Classifier.yml";
        }  

        //new variables
        string dataset_desc_file = basename + '/' + config_path;
        if(!parser.check()){
            parser.printErrors();
            return 0;
        }
        vector<string> categories;
    	vector<int> samples_per_cat;
        vector<float> rRates(r_try, 0.0);
        vector<Mat> X_train; //Keeps the feature of the images
        vector<int> y_train;//Keeps the ID of the image's category
        vector<Mat> X_test; //Keeps the feature of the images
        vector<vector<int>> train_samples; //Get the id for the images
        vector<vector<int>> test_samples;
        vector<float>labels; // Keeps the responses of the cluster's centers
        vector<Mat> BoVW;
        vector<float> true_labels; //
        vector<Mat> BoVWTest;
        
        BoVWTest.clear();
        X_train.clear();
        y_train.clear();
        X_test.clear();


        Mat matDescriptors;
        Mat auxMat; //Kmean useless matrix
        Mat trainData;//Visual word
        Mat bovw;
        Mat test_bovw;
        Mat predicted_labels;
        Mat confusion_mat;
        
        TermCriteria criteria;

        Ptr<cv::ml::KNearest> KNN = cv::ml::KNearest::create();
        Ptr<cv::ml::KNearest> KNN_Result = cv::ml::KNearest::create();
        Ptr<cv::ml::RTrees> RFTDictionary = cv::ml::RTrees::create();
        Ptr<cv::ml::RTrees> RFTResult = cv::ml::RTrees::create();

        Ptr<cv::ml::Boost> BoostDictionary = cv::ml::Boost::create();
        Ptr<cv::ml::Boost> BoostResult = cv::ml::Boost::create();
        
        double rRate_mean, rRate_dev, rRate_best=0;
        
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
   

    // Repeat training+test
    for (int trial=0; trial<r_try; trial++)
    {
        clog << "######### TRIAL " << trial+1 << " ##########" << endl;
        train_samples.clear();
        test_samples.clear();

        create_train_test_datasets(samples_per_cat, ntrain, ntest, train_samples, test_samples);

		//-----------------------------------------------------
		//                  TRAINING
		//-----------------------------------------------------

        clog << "Training ..." << endl;
        clog << "\tCreating dictionary ... " << endl;
        clog << "\t\tComputing descriptors..." << endl;

        

        //getDescriptor(train_samples, basename, , matDescriptors, X_train, ndesc, categories, false, aux);
        for (size_t c = 0; c < train_samples.size(); ++c)
        {
            clog << "  " << setfill(' ') << setw(3) << (c * 100) / train_samples.size() << " %   \015";
            for (size_t s = 0; s < train_samples[c].size(); ++s)
            {
                string filename = compute_sample_filename(basename, categories[c], train_samples[c][s]);
                Mat img = imread(filename, IMREAD_GRAYSCALE);
                y_train.push_back(c);
                if (img.empty())
                {   
                    cerr << "Error: could not read image '" << filename << "'." << endl;
                    exit(-1);
                }
                else
                {

                    // Fix size:  width =300px
                    //! \todo TO BE COMPLETED
                    Size descripsize=Size(IMG_WIDTH, (IMG_WIDTH*img.rows)/img.cols);
                    resize(img, img, descripsize);
                    // Extract SIFT descriptors
                    //-----------------------------------------------------
                    Mat descs;
                    descs = extractSIFTDescriptors(img, ndesc);
                    X_train.push_back(descs);
                    //Concatenate the images 
                    if(matDescriptors.rows!=0){
                        vconcat(matDescriptors, descs, matDescriptors);
                    }
                    else{
                        matDescriptors = descs;
                    }                    
                }                      
            }
        }
        clog << endl;
        clog << "\tGenerating " << keywords << " keywords ..." << endl;
        
        //we keep in kyws the cluster center
        kmeans(matDescriptors, keywords, auxMat, criteria, NNnumber, KMEANS_RANDOM_CENTERS, trainData);
        
        for (int i = 0; i < trainData.rows; ++i)
        {
            labels.push_back(i);
        }

        clog << "\tDictionary generated"<< endl;
        switch(classMode){
            case 1:
                KNN->train(trainData, cv::ml::ROW_SAMPLE, labels);
            break;
            case 2:
                RFTDictionary->train(trainData, cv::ml::ROW_SAMPLE, labels);
            break;
            case 3:
                BoostDictionary->train(trainData, 0, labels);
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
        
        
        labels.clear();

        // Computing the bovw
        //-----------------------------------------------------
        clog << "\tComputing BoVW ... " << endl;
        clog << "\t\tGenerating a bovw descriptor per training image." << endl;
        switch(classMode){
                case 1:
                for (int i = 0; i < X_train.size(); ++i)
                {
                    BoVW.push_back(compute_bovw(KNN, keywords, X_train[i]));
                
                }break;
                case 2:
                for (int i = 0; i < X_train.size(); ++i)
                {
                    BoVW.push_back(compute_bovw(RFTDictionary, keywords, X_train[i]));
                }
                break;
                default:
                    cout<<"Not classifier recognized"<<endl;
                    exit(-1);
                break;
        }

        // Define the classifier type and train it
		//-----------------------------------------------------
		vconcat(BoVW, bovw);
        BoVW.clear(); 
        //bovw.convertTo(bovw, CV_32F);

        clog << "\tThere are " << bovw.rows << " train images." << endl;
        switch(classMode){
            case 1:
                clog<<"\tTraining KNN..."<<endl;
                KNN_Result->train(bovw, cv::ml::ROW_SAMPLE, y_train);
                break;
            case 2:
                clog<<"\tTraining Random Forest..."<<endl;
                RFTResult->train(bovw, cv::ml::ROW_SAMPLE, y_train);
                break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
        

		//-----------------------------------------------------
		//                  TESTING
		//-----------------------------------------------------


        clog << "Testing .... " << endl;
        // First we get the descriptor of each image
        //-----------------------------------------------------
        //test_samples=train_samples;
        for (size_t c = 0; c < test_samples.size(); ++c)
        {
            clog << "  " << setfill(' ') << setw(3) << (c * 100) / test_samples.size() << " %   \015";
            for (size_t s = 0; s < test_samples[c].size(); ++s)
            {
                string filename = compute_sample_filename(basename, categories[c], test_samples[c][s]);
                Mat img = imread(filename, IMREAD_GRAYSCALE);
                true_labels.push_back(c);
                if (img.empty())
                {
                    cerr << "Error: could not read image '" << filename << "'." << endl;
                    exit(-1);
                }
                else
                {
                    Size descripsize=Size(IMG_WIDTH, (IMG_WIDTH*img.rows)/img.cols);
                    resize(img, img, descripsize);
                    // Extract SIFT descriptors
                    //-----------------------------------------------------
                    Mat descs;
                    descs = extractSIFTDescriptors(img, ndesc);
                    X_test.push_back(descs); 
                }                 
            }
        }

        //For each test image, compute the corresponding bovw.
        clog << "\tCompute image descriptors for test images..." << endl;
        switch(classMode){
            case 1:
            for (int i = 0; i < X_test.size(); ++i)
            {
                BoVWTest.push_back(compute_bovw(KNN, keywords, X_test[i]));
            
            }break;
            case 2:
            for (int i = 0; i < X_test.size(); ++i)
            {
                BoVWTest.push_back(compute_bovw(RFTDictionary, keywords, X_test[i]));
            }
            break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
        vconcat(BoVWTest, test_bovw);
        BoVWTest.clear();
        clog << "\tThere are " << test_bovw.rows << " test images." << endl;

        //Classify the test samples.
        clog << "\tClassifing test images." << endl;
        
        switch(classMode){
            case 1:
                KNN_Result->findNearest(test_bovw, NNnumber, predicted_labels);
                break;
            case 2:
                RFTResult->predict(test_bovw, predicted_labels);
                break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;
        }
        

        //compute the classifier's confusion matrix.
        clog << "\tComputing confusion matrix." << endl;
        confusion_mat = compute_confusion_matrix(categories.size(), Mat(true_labels), predicted_labels);
        CV_Assert(int(sum(confusion_mat)[0]) == test_bovw.rows);
        compute_recognition_rate(confusion_mat, rRate_mean, rRate_dev);
        cerr << "Recognition rate mean = " << rRate_mean * 100 << "% dev " << rRate_dev * 100 << endl;
        rRates[trial]=rRate_mean;

        X_train.clear();
        X_test.clear();
        true_labels.clear();
        y_train.clear();
        if(rRate_mean > rRate_best){
            rRate_best = rRate_mean;
            
            switch(classMode){
            case 1:
                clog << "\t\tSaving the best result..." << endl;
                KNN->save(dictName);
                KNN_Result->save(className);
                break;
            case 2:
                clog << "\t\tSaving the best result..." << endl;
                RFTDictionary->save(dictName);
                RFTResult->save(className);
                break;
            default:
                cout<<"Not classifier recognized"<<endl;
                exit(-1);
            break;

            }        
        }
    }

    //Saving the best models: dictionary and classifier, format YML
    //! \todo TO BE COMPLETED
  
	
    clog << "###################### FINAL STATISTICS  ################################" << endl;

    rRate_mean = 0.0;
    rRate_dev = 0.0;

	//! \todo TO BE COMPLETED
    for (int i = 0; i < rRates.size(); ++i)
    {
        rRate_mean+=rRates[i];
    }
    rRate_mean=rRate_mean/rRates.size();
    for (int i = 0; i < rRates.size(); ++i)
    {
        rRate_dev+=pow(rRates[i]-rRate_mean, 2);
    }
    if(r_try>1){
        rRate_dev = sqrt(rRate_dev/(r_try-1));
    }    
    clog << "Recognition Rate mean " << rRate_mean*100.0 << "% dev " << rRate_dev*100 << endl;
    }
    catch (exception& e)
    {
        cerr << "Capturada excepcion: " << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
}