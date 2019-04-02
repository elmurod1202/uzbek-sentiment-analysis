# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset for Uzbek language that has been manually collected from top 100 Uzbek aaps at Google play Store also adding 20K translated from english


import sys
import os
import time

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# method to show you the right thing to write to the command line when unnecessary args included:
def usage():
    print("Usage:")
    print("python %s" % sys.argv[0])

# method to count how many lines in the review file:
def count_reviews(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), 'r') as infile:
	num_lines = sum(1 for line in infile)
    return num_lines

# main method:
if __name__ == '__main__':

    # when there are unnuecessary arg in the command line:
    if len(sys.argv) > 2:
        usage()
        sys.exit(1)

    # name of the folder that input data contains:
    data_dir = "data"
    # names of classes to analyse:
    classes = ['pos', 'neg']
    # names of the input files for each class:
    filenames1 = ["UZ_positive.txt" , "UZ_negative.txt"]
    filenames2 = ["positive10kUZ.txt" , "negative10kUZ.txt"]
    # numbers of reviews for each input file:
    review_counts1 = [count_reviews(data_dir, filenames1[0]) , count_reviews(data_dir, filenames1[1])]
    review_counts2 = [count_reviews(data_dir, filenames2[0]) , count_reviews(data_dir, filenames2[1])]
    # how many % of the input data to be used for the training (recommended: 80%):
    split_percentage=80


    # Initial variables to be used:
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Counting all the input data
    count_combo_train=0;
    count_combo_test=0;

    # Read the dataset1: Annotated dataeset
    print "\n### Reading the annotated dataset:" 
    sum_count_train=0;
    sum_count_test=0;
    for polarity in range(2):
	    print "Polarity class: "+classes[polarity]+ ", Opening file: "+filenames1[polarity]+":\n"
	    with open(os.path.join(data_dir, filenames1[polarity]), 'r') as infile:
		count=0
		count_train=0
		count_test=0
		for line in infile.readlines():
		    count+=1
		    review=line.strip('\n')
		    # some % of the input data will be used to test:
		    if count>(float(split_percentage)/100)*review_counts1[polarity]:
			count_test+=1
			test_data.append(review)
			test_labels.append(classes[polarity])
		    	#print "Review N: "+str(count)+", test data, comment: "+review
		    else:
			count_train+=1
			train_data.append(review)
			train_labels.append(classes[polarity])
		    	#print "Review N: "+str(count)+", train data, comment: "+review
		print "Number of reveiws: "+str(count)+", training data: "+str(count_train)+", test data: "+str(count_test)+"\n"
		sum_count_train+=count_train
		sum_count_test+=count_test
    count_combo_train+=sum_count_train
    count_combo_test+=sum_count_test

    #reporting the read data:
    print "Total number of reveiws: "+str(sum_count_train+sum_count_test)+", total training data: "+str(sum_count_train)+", total test data: "+str(sum_count_test)+"\n"


    # Read the dataset2: Translated dataeset
    print "\n### Reading the translated dataset:" 
    sum_count_train=0;
    sum_count_test=0;
    for polarity in range(2):
	    print "Polarity class: "+classes[polarity]+ ", Opening file: "+filenames2[polarity]+":\n"
	    with open(os.path.join(data_dir, filenames2[polarity]), 'r') as infile:
		count=0
		count_train=0
		count_test=0
		for line in infile.readlines():
		    count+=1
		    review=line.strip('\n')
		    # some % of the input data will be used to test:
		    if count>(float(split_percentage)/100)*review_counts2[polarity]:
			count_test+=1
			test_data.append(review)
			test_labels.append(classes[polarity])
		    	#print "Review N: "+str(count)+", test data, comment: "+review
		    else:
			count_train+=1
			train_data.append(review)
			train_labels.append(classes[polarity])
		    	#print "Review N: "+str(count)+", train data, comment: "+review
		print "Number of reveiws: "+str(count)+", training data: "+str(count_train)+", test data: "+str(count_test)+"\n"
		sum_count_train+=count_train
		sum_count_test+=count_test
    count_combo_train+=sum_count_train
    count_combo_test+=sum_count_test

    #reporting the read data:
    print "Total number of reveiws: "+str(sum_count_train+sum_count_test)+", total training data: "+str(sum_count_train)+", total test data: "+str(sum_count_test)+"\n"
    print "\n### Both Datasets: "+str(count_combo_train+count_combo_test)+", total training data: "+str(count_combo_train)+", total test data: "+str(count_combo_test)+"\n"


    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC(gamma='auto')
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', gamma='auto')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print(accuracy_score(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print(accuracy_score(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
    print(accuracy_score(test_labels, prediction_liblinear))



