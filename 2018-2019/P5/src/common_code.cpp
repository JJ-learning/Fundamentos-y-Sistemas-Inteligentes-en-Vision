/*! \file common_code.cpp
    \brief Useful for building a Bag of Visual Words model
    \authors Fundamentos de Sistemas Inteligentes en Vision
*/

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

void basename(const string &path,
              string &dirname,
              string &filename,
              string &ext)
{
    dirname = "";
    filename = path;
    ext = "";

    auto pos = path.rfind("/");

    if (pos != string::npos)
    {
        dirname = path.substr(0, pos);
        filename = path.substr(pos + 1);
    }

    pos = filename.rfind(".");

    if (pos != string::npos)
    {
        ext = filename.substr(pos + 1);
        filename = filename.substr(0, pos);
    }
    return;
}

string compute_sample_filename(const string &basename, const string &cat, const int sample_index)
{
    ostringstream filename;
    filename << basename << "" << cat << "/image_" << setfill('0') << setw(4) << sample_index << ".jpg";
    return filename.str();
}

int load_dataset_information(const string &fname, vector<string> &categories, vector<int> &samples_per_cat)
{
    int retCode = 0;
    ifstream in(fname);

    if (!in)
        retCode = 1;
    else
    {
        while ((in) && (retCode == 0))
        {
            string catName;
            int nsamples;
            in >> catName >> nsamples;
            if (!in)
            {
                if (!in.eof())
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

void random_sampling(int total, int ntrain, int ntest,
                     vector<int> &train_samples,
                     vector<int> &test_samples)
{
    assert(ntrain < total);
    train_samples.resize(0);
    test_samples.resize(0);
    vector<bool> sampled(total, false);
    while (ntrain > 0)
    {
        int s = int(double(total) * rand() / (RAND_MAX + 1.0));
        int i = 0;
        while (sampled[i] && i < sampled.size())
            ++i; //the first unsampled.
        int c = 0;
        while (c < s) //count s unsampled.
        {
            while (sampled[++i])
                ; //advance to next unsampled.
            ++c;
        }
        assert(!sampled[i]);
        train_samples.push_back(i + 1);
        sampled[i] = true;
        --total;
        --ntrain;
    }
    if (ntest >= total)
    {
        for (size_t i = 0; i < sampled.size(); ++i)
            if (!sampled[i])
                test_samples.push_back(i + 1);
    }
    else
    {
        while (ntest > 0)
        {
            int s = int(double(total) * rand() / (RAND_MAX + 1.0));
            int i = 0;
            while (sampled[i] && i < sampled.size())
                ++i; //the first unsampled.
            int c = 0;
            while (c < s) //count s unsampled.
            {
                while (sampled[++i])
                    ; //advance to next unsampled.
                ++c;
            }
            test_samples.push_back(i + 1);
            sampled[i] = true;
            --total;
            --ntest;
        }
    }
}

void create_train_test_datasets(vector<int> &samples_per_cat, int ntrain_samples, int ntest_samples,
                                vector<vector<int>> &train_samples, vector<vector<int>> &test_samples)
{
    train_samples.resize(0);
    test_samples.resize(0);
    for (size_t i = 0; i < samples_per_cat.size(); ++i)
    {
        vector<int> train;
        vector<int> test;
        random_sampling(samples_per_cat[i], ntrain_samples, ntest_samples, train, test);
        train_samples.push_back(train);
        test_samples.push_back(test);
    }
}

Mat compute_confusion_matrix(int n_categories, const Mat &true_labels, const Mat &predicted_labels)
{
    CV_Assert(true_labels.rows == predicted_labels.rows);
    CV_Assert(true_labels.type() == CV_32FC1);
    CV_Assert(predicted_labels.type() == CV_32FC1);
    Mat confussion_mat = Mat::zeros(n_categories, n_categories, CV_32F);
    for (int i = 0; i < true_labels.rows; ++i)
    {
        confussion_mat.at<float>(true_labels.at<float>(i), predicted_labels.at<float>(i)) += 1.0;
    }

    cout << confussion_mat << endl;
    return confussion_mat;
}

void compute_recognition_rate(const Mat &cmat, double &mean, double &dev)
{
    CV_Assert(cmat.rows == cmat.cols && cmat.rows > 1);
    CV_Assert(cmat.depth() == CV_32F);

    mean = 0.0;
    dev = 0.0;
    for (int c = 0; c < cmat.rows; ++c)
    {
        const double class_Rate = cmat.at<float>(c, c) / sum(cmat.row(c))[0];
        mean += class_Rate;
        dev += class_Rate * class_Rate;
    }
    mean /= double(cmat.rows);
    dev = sqrt(dev / double(cmat.rows) - mean * mean);
}

Mat extractSIFTDescriptors(const Mat &img, const int ndesc)
{
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(ndesc);
    vector<KeyPoint> kps;
    Mat descs;
    sift->detectAndCompute(img, noArray(), kps, descs);
    return descs;
}

Mat compute_bovw(const Ptr<ml::KNearest> &dict, const int dict_size, Mat &img_descs, bool normalize)
{
    Mat bovw = Mat::zeros(1, dict_size, CV_32F);
    Mat vwords;
    CV_Assert(img_descs.type() == CV_32F);
    dict->findNearest(img_descs, 10, vwords);
    CV_Assert(vwords.depth() == CV_32F);
    for (int i = 0; i < img_descs.rows; ++i)
        bovw.at<float>(vwords.at<float>(i))++;
    if (normalize)
        bovw /= float(img_descs.rows);
    return bovw;
}

Mat extractSURFdescriptors(const Mat &img, const int threshold)
{
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(threshold, 4, 3, true);
    vector<KeyPoint> kps;
    Mat descs;
    surf->detectAndCompute(img, noArray(), kps, descs);
    return descs;
}

Mat extractDenseSIFTdescriptors(const Mat &img, const int ndesc, const int step)
{
    vector<KeyPoint> keypoints;
    for (int y=step; y<img.rows-step; y+=step){
        for (int x=step; x<img.cols-step; x+=step){

            // x,y,radius
            keypoints.push_back(KeyPoint(float(x), float(y), float(step)));
        }
    }
    Mat descs;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(ndesc);
    sift->compute(img, keypoints, descs);
    return descs;
}
