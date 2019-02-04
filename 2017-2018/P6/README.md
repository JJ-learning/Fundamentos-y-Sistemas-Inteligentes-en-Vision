# HOW TO COMPILE AND RUN THE PROGRAM

## 1.In order to get the executables run the following command:
	> cd built 
	> cmake ..
	> make
## 2.In order to get the classifier and the dictionary, we have to run:
	### Run by default:	
		 > ./train_bovw -b=../data -c=101_ObjectCategories/configfiles/02_ObjectCategories_conf.txt
	### To know the argument nampossibilities:	
		> ./train_bovw -h
## 3.To run the test to get the category of an image
	### Run By default:	
		> ./test_bovw -b=../data -c=101_ObjectCategories/configfiles/02_ObjectCategories_conf.txt -fileName=[Name of the image]
	### To know the argument possibilities	
	> ./test_bovw -h
