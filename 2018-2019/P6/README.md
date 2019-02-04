# Assignment 6 of FSIV course

In this assignment I've done all the mandaroty part and the first additional part as well. In order to compile the program, the user just have to run the ./run.sh script to make an cmake and make.  We can see how to run the program within the following paragraphs.

In the terminal, run the script ./run.sh in order to get the executable, later on, once compiled, you have to run the following command in build folder:
>  ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=knn -n_k=2 -desc_t=1
>  ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -n_c=2 -svm_type=polynomial -desc_t=0
>  ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=boosting -n_b=1 -desc_t=1

In order to run the test, we have to run the following command:
> ./test_bovw -dict=../model/dictionary.yml -classifier=../models/classifier.yml -config_file=../models/02_ObjectCategories_conf.txt -img=../data/airplane1.jpg 
> ./test_bovw -dict=../models/dictionary_KNN.yml -classifier=../models/classifier_KNN.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=0 -vid=../video_image_categ_part2.avi 

### Possible ways to run the train_bovw executable:
#### SIFT descriptor:
##### Train:
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=knn -n_k=2 -desc_t=0 -keywords=150

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=radial -n_c=1 -desc_t=0 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=polynomial -n_c=2 -desc_t=0 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=linear -n_c=2 -desc_t=0 -keywords=100

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=boosting -n_b=200 -desc_t=0 -keywords=100

##### Test:
> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_KNN.yml  -classifier=../models/classifier_02_SIFT_KNN.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=0 -vid=../video_image_categ_part2.avi -class_t=knn -n_k=2

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_SVM.yml  -classifier=../models/classifier_02_SIFT_SVM.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=0 -vid=../video_image_categ_part2.avi -class_type=svm -n_c=2 -svm_type=radial 

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_Boosting.yml  -classifier=../models/classifier_02_SIFT_Boosting.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=0 -vid=../video_image_categ_part2.avi -class_type=boosting -n_b=300

#### SURF descriptor:
##### Train:
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=knn -n_k=2 -desc_t=1 -keywords=150

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=radial -n_c=1 -desc_t=1 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=polynomial -n_c=2 -desc_t=1 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=linear -n_c=2 -desc_t=1 -keywords=100

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=boosting -n_b=200 -desc_t=1 -keywords=100

##### Test:
> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_KNN.yml  -classifier=../models/classifier_02_SIFT_KNN.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=1 -vid=../video_image_categ_part2.avi -class_t=knn -n_k=2

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_SVM.yml  -classifier=../models/classifier_02_SIFT_SVM.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=1 -vid=../video_image_categ_part2.avi -class_type=svm -n_c=2 -svm_type=radial 

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_Boosting.yml  -classifier=../models/classifier_02_SIFT_Boosting.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=1 -vid=../video_image_categ_part2.avi -class_type=boosting -n_b=300

#### Dense SIFT descriptor:
##### Train:
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=knn -n_k=2 -desc_t=2 -keywords=150

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=radial -n_c=1 -desc_t=2 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=polynomial -n_c=2 -desc_t=2 -keywords=100
> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=svm -svm_type=linear -n_c=2 -desc_t=2 -keywords=100

> ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt -class_type=boosting -n_b=200 -desc_t=2 -keywords=100

##### Test:
> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_KNN.yml  -classifier=../models/classifier_02_SIFT_KNN.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=2 -vid=../video_image_categ_part2.avi -class_t=knn -n_k=2

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_SVM.yml  -classifier=../models/classifier_02_SIFT_SVM.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=2 -vid=../video_image_categ_part2.avi -class_type=svm -n_c=2 -svm_type=radial 

> ./test_bovw -dict=/Users/juanjo/code/Uni/FSIV/Practicas/P6/models/dictionary_02_SIFT_Boosting.yml  -classifier=../models/classifier_02_SIFT_Boosting.yml -config_file=../models/02_ObjectCategories_conf.txt  -desc_t=2 -vid=../video_image_categ_part2.avi -class_type=boosting -n_b=300