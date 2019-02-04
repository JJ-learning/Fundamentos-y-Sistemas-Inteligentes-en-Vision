They way that we train the data by default should be like:
	./train_bovw -b=../dataset -c=101_ObjectCategories/configfiles/02_ObjectCategories_conf.txt -r=3

The way we execute the test should looks like:
	./test_bovw -b=../dataset -c=101_ObjectCategories/configfiles/02_ObjectCategories_conf.txt -f=airplane.jpg