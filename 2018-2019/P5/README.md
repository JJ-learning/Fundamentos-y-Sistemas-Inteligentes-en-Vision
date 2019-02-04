# How to compile the program
In the terminal, run the script ./run.sh in order to get the executable, later on, once compiled, you have to run the following command in build folder:
>  ./train_bovw -b=../data/101_ObjectCategories/ -c=../models/02_ObjectCategories_conf.txt 

In order to run the test, we have to run the following command:
> ./test_bovw -dict=../model/cat2_N10_dictionary.yml -classifier=../models/cat2_N10_classifier_KNN10.yml -config_file=../models/02_ObjectCategories_conf.txt -img=../data/airplane1.jpg 


Nota: Las parted adicionales que he hecho han sido, añadir el descriptor Dense SIFT y además los resultados gráficos, es decir, cuando ejecutamos el test, encima de la imagen original aparece la clase predicha.