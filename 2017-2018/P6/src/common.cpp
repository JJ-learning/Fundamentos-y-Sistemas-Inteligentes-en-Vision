#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include "common.hpp"
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

void basename(const string& path, string& dirname, string& filename,string& ext)
{
    dirname="";
    filename=path;
    ext="";

    auto pos = path.rfind("/");

    if (pos != string::npos)
    {
      dirname=path.substr(0, pos);
      filename=path.substr(pos+1);
    }

    pos = filename.rfind(".");

    if (pos != string::npos)
    {
      ext = filename.substr(pos+1);
      filename = filename.substr(0,pos);
    }
    return;
}

string compute_sample_filename(const string& basename, const string& cat, const int sample_index)
{
    ostringstream filename;
    filename << basename << "/101_ObjectCategories/" << cat << "/image_" << setfill('0') << setw(4) << sample_index << ".jpg";
    return filename.str();
}

int load_dataset_information(const string& fname, vector<string>& categories, vector<int>& samples_per_cat)
{
   int retCode = 0;
   ifstream in (fname);

   if (!in)
       retCode = 1;
   else
   {
       while((in) && (retCode==0) )
       {
           string catName;
           int nsamples;
           in >> catName >> nsamples;
           if (!in)
           {
               if (! in.eof())
                retCode = 2;
           }
           else
           {
               categories.push_back(catName);
               samples_per_cat.push_back(nsamples);
           }
       }
   }
   return retCode;
}

void random_sampling (int total, int ntrain, int ntest, vector< int >& train_samples, vector< int >& test_samples)
{
    assert(ntrain<total);
    train_samples.resize(0);
    test_samples.resize(0);
    vector<bool> sampled(total, false);
    while (ntrain>0)
    {
        int s = int(double(total) * rand()/(RAND_MAX+1.0));
        int i=0;
        while(sampled[i] && i<sampled.size()) ++i; //the first unsampled.
        int c=0;
        while (c<s) //count s unsampled.
        {
            while (sampled[++i]); //advance to next unsampled.
            ++c;
        }
        assert(!sampled[i]);
        train_samples.push_back(i+1);
        sampled[i]=true;
        --total;
        --ntrain;
    }
    if (ntest>=total)
    {
        for (size_t i=0 ; i<sampled.size(); ++i)
            if (!sampled[i])
                test_samples.push_back(i+1);
    }
    else
    {
        while (ntest>0)
        {
            int s = int(double(total) * rand()/(RAND_MAX+1.0));
            int i=0;
            while(sampled[i] && i<sampled.size()) ++i; //the first unsampled.
            int c=0;
            while (c<s) //count s unsampled.
            {
                while (sampled[++i]); //advance to next unsampled.
                ++c;
            }
            test_samples.push_back(i+1);
            sampled[i]=true;
            --total;
            --ntest;
        }
    }
}

void create_train_test_datasets (vector<int>& samples_per_cat, int ntrain_samples, int ntest_samples, vector< vector <int> >& train_samples, vector< vector<int> >& test_samples)
{
    train_samples.resize(0);
    test_samples.resize(0);
    for (size_t i=0;i<samples_per_cat.size(); ++i)
    {
        vector<int> train;
        vector<int> test; 
        random_sampling (samples_per_cat[i], ntrain_samples, ntest_samples, train, test);
        train_samples.push_back(train);
        test_samples.push_back(test);
    }
}

Mat compute_confusion_matrix(int n_categories, const Mat& true_labels, const Mat& predicted_labels)
{
    CV_Assert(true_labels.rows == predicted_labels.rows);
    CV_Assert(true_labels.type()==CV_32FC1);
    CV_Assert(predicted_labels.type()==CV_32FC1);
    Mat confussion_mat = Mat::zeros(n_categories, n_categories, CV_32F);
    for (int i = 0; i < true_labels.rows; ++i)
    {
        confussion_mat.at<float>(true_labels.at<float>(i), predicted_labels.at<float>(i)) += 1.0;
    }
    cout<<confussion_mat<<endl;
    return confussion_mat;
}

void compute_recognition_rate(const Mat& cmat, double& mean, double& dev)
{
    CV_Assert(cmat.rows == cmat.cols && cmat.rows>1);
    CV_Assert(cmat.depth()==CV_32F);

    mean = 0.0;
    dev = 0.0;
    for (int c=0; c<cmat.rows; ++c)
    {
        const double class_Rate = cmat.at<float>(c,c)/sum(cmat.row(c))[0];
        mean += class_Rate;
        dev += class_Rate*class_Rate;
    }
    mean /= double(cmat.rows);
    dev = sqrt(dev/double(cmat.rows) - mean*mean);
}

Mat extractSIFTDescriptors(const Mat& img, const int ndesc)
{
    Mat descs;
    //! \todo TO BE COMPLETED
	 //Create the SIFT variable
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(ndesc);
    vector<KeyPoint> keypoints;
    detector->detectAndCompute(img, Mat(), keypoints, descs);
    return descs;
}

/*
  dict is the bagofWords
*/
Mat compute_bovw (const Ptr<ml::KNearest>& dict, const int dict_size, Mat& img_descs, bool normalize)
{
    Mat bovw(1, dict_size, CV_32FC1, Scalar(0));
    Mat bovwresult(1, dict_size, CV_32FC1, Scalar(0));
    //! \todo TO BE COMPLETED	
    dict->findNearest(img_descs, 1, bovwresult);
    
    int aux;
    for (int i = 0; i < bovwresult.rows; ++i)
    {
      aux=(int)bovwresult.at<float>(i);
      bovw.at<float>(aux)++;
    }
    if(normalize==true){
      for (int i = 0; i < bovw.cols; ++i)
      {
        for(int j = 0; j< bovw.rows; j++){
          bovw.at<float>(j,i) = bovw.at<float>(j,i)/bovw.rows;
        }
      }
    }   
	 
    return bovw;
}

Mat compute_bovw (const Ptr<ml::RTrees>& dict, const int dict_size, Mat& img_descs, bool normalize)
{
    Mat bovw(1, dict_size, CV_32FC1, Scalar(0));
    Mat bovwresult(1, dict_size, CV_32FC1, Scalar(0));
    //! \todo TO BE COMPLETED 
    dict->predict(img_descs, bovwresult);
    
    int aux;
    for (int i = 0; i < bovwresult.rows; ++i)
    {
      aux=(int)bovwresult.at<float>(i);
      bovw.at<float>(aux)++;
    }
    if(normalize==true){
      for (int i = 0; i < bovw.cols; ++i)
      {
        for(int j = 0; j< bovw.rows; j++){
          bovw.at<float>(j,i) = bovw.at<float>(j,i)/bovw.rows;
        }
      }
    }   
   
    return bovw;
}

Mat compute_bovw (const Ptr<ml::Boost>& dict, const int dict_size, Mat& img_descs, bool normalize)
{
    Mat bovw(1, dict_size, CV_32FC1, Scalar(0));
    Mat bovwresult(1, dict_size, CV_32FC1, Scalar(0));
    //! \todo TO BE COMPLETED 
    dict->predict(img_descs, bovwresult);
    
    int aux;
    for (int i = 0; i < bovwresult.rows; ++i)
    {
      aux=(int)bovwresult.at<float>(i);
      bovw.at<float>(aux)++;
    }
    if(normalize==true){
      for (int i = 0; i < bovw.cols; ++i)
      {
        for(int j = 0; j< bovw.rows; j++){
          bovw.at<float>(j,i) = bovw.at<float>(j,i)/bovw.rows;
        }
      }
    }   
   
    return bovw;
}
