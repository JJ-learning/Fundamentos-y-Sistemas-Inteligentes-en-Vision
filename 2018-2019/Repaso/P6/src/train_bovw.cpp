#include "common_code.hpp"

#define IMG_WIDTH 300

using namespace std;
using namespace cv;

const string keys =
    "{help h usage ? |      | print this message   }"
    "{b              | ./data   | basename for the dataset      }"
    "{c              |     | configuration file for the dataset      }"
    "{r               | 10 | Number of trials train/test to comute the reconition rate       }"
    "{keywords               | 100 | [Kmeans] Number of keywords generated        }"
    "{ntrain            | 15 | Number of samples per class used to train }"
    "{ntest            | 50 | number of samples per class used to test}"
    "{ndesc            | 0 | [SIFT] number of descriptor per image. Value 0 means extract all }"
    "{dict_runs            | 5 |[SIFT] Number of trials to select the best dictionary.}"
    "{desc_t            | 0 | type of descriptor used to train the data set. Default: 0(SIFT). 0->SIFT. 1->SURF. 2->Dense SIFT}"
    "{threshold            | 400 | threshold used in SURF. Default: 400}"
    "{steps            | 10 | distance from where you take the keypoint(Dense SIFT).}"
    "{n_k            | 1 | Number neightbours used in knn}"
    "{n_c            | 1 | parameter C of the svm}"
    "{n_b            | 100 | Number of weak counts [Boosting]}"
    "{class_type            | knn | type of the classifier used}"
    "{svm_type            | linear | kernel used in svm}";

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {
        CommandLineParser parser(argc, argv, keys);
        parser.about("Practica 5. Categorizacion de imagenes.");

        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
        // Parser variables
        string base_name = parser.get<string>("b");
        string config_file = parser.get<string>("c");

        int n_runsArg = parser.get<int>("r");
        int ntrain = parser.get<int>("ntrain");
        int ntest = parser.get<int>("ntest");
        int ndescs = parser.get<int>("ndesc");
        int keywords = parser.get<int>("keywords");
        int dict_runs = parser.get<int>("dict_runs");
        int desc_t = parser.get<int>("desc_t");
        int threshold = parser.get<int>("threshold");
        int steps = parser.get<int>("steps");
        int c_value = parser.get<int>("n_c");
        int b_value = parser.get<int>("n_b");
        int k_value = parser.get<int>("n_k");

        string classifier_type = parser.get<string>("class_type");
        string svm_kernel = parser.get<string>("svm_type");

        // New variables
        vector<string> categories;            // Keeps all the categories
        vector<int> samples_per_cat;          // Keeps the number of samples per category
        vector<float> rRates(n_runsArg, 0.0); //
        vector<int> siftScales{9, 13};        // 5, 9

        int sift_type = 0;

        Ptr<ml::KNearest> best_dictionary;
        Ptr<ml::StatModel> best_classifier;

        double best_rRate = 0.0;

        string dataset_desc_file = config_file;

        if ((retCode = load_dataset_information(dataset_desc_file, categories, samples_per_cat)) != 0)
        {
            cerr << "Error: could not load dataset information from '"
                 << dataset_desc_file
                 << "' (" << retCode << ")." << endl;
            exit(-1);
        }

        cout << "Found " << categories.size() << " categories: ";
        if (categories.size() < 2)
        {
            cerr << "Error: at least two categories are needed" << endl;
            return -1;
        }

        for (size_t i = 0; i < categories.size(); i++)
        {
            cout << categories[i] << " ";
        }
        cout << endl;

        for (int trial = 0; trial < n_runsArg; trial++)
        {
            clog << "######### TRIAL " << trial + 1 << " ##########" << endl;

            vector<vector<int>> train_samples; // Keeps the samples for the train set
            vector<vector<int>> test_samples;  // Keeps the samples for the test set

            create_train_test_datasets(samples_per_cat, ntrain, ntest, train_samples, test_samples);

            clog << "Training..." << endl;
            clog << "\tCreating dictionary..." << endl;
            clog << "\t\tComputing descriptors..." << endl;

            Mat train_descs; // Keeps the descriptors of the train set
            vector<int> ndescs_per_samples;
            ndescs_per_samples.resize(0);

            for (size_t c = 0; c < train_samples.size(); c++)
            {
                clog << " " << setfill(' ') << setw(3) << (c * 100) / train_samples.size() << "% \015";
                for (size_t s = 0; s < train_samples[c].size(); s++)
                {
                    string filename = compute_sample_filename(base_name, categories[c], train_samples[c][s]);
                    Mat img = imread(filename, IMREAD_GRAYSCALE);

                    if (img.empty())
                    {
                        cerr << "Error: could not read image '" << filename << "'." << endl;
                        exit(-1);
                    }
                    else
                    {
                        // Fix size
                        resize(img, img, Size(IMG_WIDTH, round(IMG_WIDTH * img.rows / img.cols)));

                        Mat descs;
                        if (desc_t == 0)
                        {
                            descs = extractSIFTDescriptors(img, ndescs);
                        }
                        else if (desc_t == 1)
                        {
                            descs = extractSURFdescriptors(img, threshold);
                        }
                        else
                        {
                            descs = extractDenseSIFTdescriptors(img, ndescs, steps);
                        }

                        if (train_descs.empty())
                        {
                            train_descs = descs;
                        }
                        else
                        {
                            Mat dst;
                            vconcat(train_descs, descs, dst);
                            train_descs = dst;
                        }

                        ndescs_per_samples.push_back(descs.rows); // we could really have less of wished descriptors.
                    }
                }
            }
            clog << endl;
            CV_Assert(ndescs_per_samples.size() == (categories.size() * ntrain));
            clog << "\t\tDescriptors size = " << train_descs.rows * train_descs.cols * sizeof(float) / (1024.0 * 1024.0) << " MiB." << endl;
            clog << "\tGenerating " << keywords << " keywords ..." << endl;
            Mat keyws;
            Mat labels;
            double compactness = kmeans(train_descs, keywords, labels,
                                        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
                                        dict_runs,
                                        KmeansFlags::KMEANS_PP_CENTERS, //KMEANS_RANDOM_CENTERS,
                                        keyws);

            CV_Assert(keywords == keyws.rows);
            //free not needed memory
            labels.release();

            clog << "\tGenerating the dictionary ... " << endl;
            Ptr<ml::KNearest> dict = ml::KNearest::create();
            dict->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
            dict->setIsClassifier(true);
            Mat indexes(keyws.rows, 1, CV_32S);
            for (int i = 0; i < keyws.rows; ++i)
            {
                indexes.at<int>(i) = i;
            }
            dict->train(keyws, ml::ROW_SAMPLE, indexes);
            clog << "\tDictionary compactness " << compactness << endl;

            clog << "\tTrain classifier ... " << endl;

            //For each train image, compute the corresponding bovw.
            clog << "\t\tGenerating the a bovw descriptor per train image." << endl;
            int row_start = 0;
            Mat train_bovw;
            vector<int> train_labels_v;
            train_labels_v.resize(0);
            for (size_t c = 0, i = 0; c < train_samples.size(); ++c)
            {
                for (size_t s = 0; s < train_samples[c].size(); ++s, ++i)
                {
                    Mat descriptors = train_descs.rowRange(row_start, row_start + ndescs_per_samples[i]);
                    row_start += ndescs_per_samples[i];
                    Mat bovw = compute_bovw(dict, keyws.rows, descriptors);
                    train_labels_v.push_back(c);
                    if (train_bovw.empty())
                        train_bovw = bovw;
                    else
                    {
                        Mat dst;
                        vconcat(train_bovw, bovw, dst);
                        train_bovw = dst;
                    }
                }
            }

            //free not needed memory
            train_descs.release();

            // Create the classifier.
            Ptr<ml::StatModel> classifier;
            if (classifier_type == "knn")
            {
                // Train a KNN classifier using the training bovws like patterns.
                clog << "The user choose KNN classifier" << endl;
                Ptr<ml::KNearest> knnClassifier = ml::KNearest::create();
                knnClassifier->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
                knnClassifier->setDefaultK(k_value);
                knnClassifier->setIsClassifier(true);
                classifier = knnClassifier;
            }
            else if (classifier_type == "svm")
            {

                clog << "The user choose SVM classifier" << endl;
                Ptr<ml::SVM> svmClassifier = ml::SVM::create();
                if (svm_kernel == "linear")
                {
                    svmClassifier->setKernel(ml::SVM::LINEAR);
                }
                else if (svm_kernel == "polynomial")
                {
                    svmClassifier->setKernel(ml::SVM::POLY);
                    svmClassifier->setDegree(1);
                }
                else if (svm_kernel == "radial")
                {
                    svmClassifier->setKernel(ml::SVM::RBF);
                }
                else
                {
                    cerr << "Not kernel identified... Exiting program..." << endl;
                    exit(-1);
                }
                svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
                svmClassifier->setType(ml::SVM::C_SVC);
                svmClassifier->setC(c_value);
                classifier = svmClassifier;
            }
            else if (classifier_type == "boosting")
            {

                clog << "The user choose Boosting classifier" << endl;
                Ptr<ml::Boost> boostingClassifier = ml::Boost::create();
                boostingClassifier->setBoostType(ml::Boost::DISCRETE);
                boostingClassifier->setWeakCount(b_value);
                classifier = boostingClassifier;
            }

            Mat train_labels(train_labels_v);
            // We train the model

            classifier->train(train_bovw, ml::ROW_SAMPLE, train_labels);

            //free not needed memory.
            train_bovw.release();
            train_labels_v.resize(0);

            clog << "Testing .... " << endl;

            //load test images, generate SIFT descriptors and quantize getting a bovw for each image.
            //classify and compute errors.

            //For each test image, compute the corresponding bovw.
            clog << "\tCompute image descriptors for test images..." << endl;
            Mat test_bovw;
            vector<float> true_labels;
            true_labels.resize(0);
            for (size_t c = 0; c < test_samples.size(); ++c)
            {
                clog << "  " << setfill(' ') << setw(3) << (c * 100) / train_samples.size() << " %   \015";
                for (size_t s = 0; s < test_samples[c].size(); ++s)
                {
                    string filename = compute_sample_filename(base_name, categories[c], test_samples[c][s]);
                    Mat img = imread(filename, IMREAD_GRAYSCALE);
                    if (img.empty())
                        cerr << "Error: could not read image '" << filename << "'." << endl;
                    else
                    {
                        // Fix size
                        resize(img, img, Size(IMG_WIDTH, round(IMG_WIDTH * img.rows / img.cols)));

                        //Mat descs = extractSIFTDescriptors(img, ndesc.getValue());
                        Mat descs;

                        if (desc_t == 0)
                        {
                            descs = extractSIFTDescriptors(img, ndescs);
                        }
                        else if (desc_t == 1)
                        {
                            descs = extractSURFdescriptors(img, threshold);
                        }
                        else
                        {
                            descs = extractDenseSIFTdescriptors(img, ndescs, steps);
                        }

                        Mat bovw = compute_bovw(dict, keyws.rows, descs);
                        if (test_bovw.empty())
                            test_bovw = bovw;
                        else
                        {
                            Mat dst;
                            vconcat(test_bovw, bovw, dst);
                            test_bovw = dst;
                        }
                        true_labels.push_back(c);
                    }
                }
            }
            clog << endl;
            clog << "\tThere are " << test_bovw.rows << " test images." << endl;

            //classify the test samples.
            clog << "\tClassifying test images." << endl;
            Mat predicted_labels;

            classifier->predict(test_bovw, predicted_labels);

            CV_Assert(predicted_labels.depth() == CV_32F);
            CV_Assert(predicted_labels.rows == test_bovw.rows);
            CV_Assert(predicted_labels.rows == true_labels.size());

            //compute the classifier's confusion matrix.
            clog << "\tComputing confusion matrix." << endl;
            Mat confusion_mat = compute_confusion_matrix(categories.size(), Mat(true_labels), predicted_labels);

            CV_Assert(int(sum(confusion_mat)[0]) == test_bovw.rows);
            double rRate_mean, rRate_dev;
            compute_recognition_rate(confusion_mat, rRate_mean, rRate_dev);
            cerr << "Recognition rate mean = " << rRate_mean * 100 << "% dev " << rRate_dev * 100 << endl;
            rRates[trial] = rRate_mean;

            if (trial == 0 || rRate_mean > best_rRate)
            {
                best_dictionary = dict;
                best_classifier = classifier;
                best_rRate = rRate_mean;
            }
        }
        //Saving the best models.
        FileStorage dictFile;
        dictFile.open("../models/dictionary.yml", FileStorage::WRITE);
        dictFile << "keywords" << keywords;
        best_dictionary->write(dictFile);
        dictFile.release();
        best_classifier->save("../models/classifier.yml");

        clog << "###################### FINAL STATISTICS  ################################" << endl;

        double rRate_mean = 0.0;
        double rRate_dev = 0.0;

        for (size_t i = 0; i < rRates.size(); ++i)
        {
            const float v = rRates[i];
            rRate_mean += v;
            rRate_dev += v * v;
        }
        rRate_mean /= double(rRates.size());
        rRate_dev = rRate_dev / double(rRates.size()) - rRate_mean * rRate_mean;
        rRate_dev = sqrt(rRate_dev);
        clog << "Recognition Rate mean " << rRate_mean * 100.0 << "% dev " << rRate_dev * 100.0 << endl;
    }
    catch (exception &e)
    {
        cerr << "Capturada excepcion" << e.what() << endl;
        retCode = EXIT_FAILURE;
    }
    return 0;
}
