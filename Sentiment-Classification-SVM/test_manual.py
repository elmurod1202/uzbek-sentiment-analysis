# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Datasets: 3 Polarity datasets for Uzbek language: manually collected ,the one translated from English and the one as their combination


import sys
import os
import time
import numpy as np

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# method to show you the right thing to write to the command line when unneccessary args included:
def usage():
    print("please, no arguments after the python file name, the sentences to test are inserted after the code runs.")
    print("Usage:")
    print("python %s" % sys.argv[0])

# method to return the proper definition for the polarity class element
def get_polarity_name(polarity):
    if polarity=="pos":
	return "POSITIVE"
    elif polarity=="neg":
	return "NEGATIVE"
    else:
	return "Unknown"

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
    #Names of the datasets to be used in order:
    datasets = ["Manual     ","Translated ","Combination"]
    # names of the input files for each class:
    filenames = [["UZ_positive.txt" , "UZ_negative.txt"], ["positive10kUZ.txt" , "negative10kUZ.txt"]]


    # Initial variables to be used:
    # So, 1(0 in array)-manually collected dataset, 2(1 in array)-the translated dataset, 3(2 in array)-the combination of both datasets
    train_datas = [[],[],[]]
    train_labels = [[],[],[]]
    counts = [[0,0],[0,0],[0,0]]

    # Read the data
    #looping through datasets: manual and translated
    for dataset_number in range(2):
	    print "Reading "+datasets[dataset_number]+" Dataset:"
	    #looping through polarity files: positives and negatives
	    for polarity in range(2):
		    print "\tPolarity class: "+get_polarity_name(classes[polarity])+ ", Opening file: "+filenames[dataset_number][polarity]
		    #opening the file:
		    with open(os.path.join(data_dir, filenames[dataset_number][polarity]), 'r') as infile:
			for line in infile.readlines():
			    #count++
			    counts[dataset_number][polarity]+=1
			    #correcting the line
			    review=line.strip('\n')
			    #adding the review to training set
			    train_datas[dataset_number].append(review)
			    #also adding it to the combination training set
			    train_datas[2].append(review)
			    #adding the label to labels set
			    train_labels[dataset_number].append(classes[polarity])
			    #also adding it to the combination labels set
			    train_labels[2].append(classes[polarity])
	    #reporting the read data:
	    print "Positive reveiws: "+str(counts[dataset_number][0])+", Negative reviews: "+str(counts[dataset_number][1])+", total: "+str(counts[dataset_number][0]+counts[dataset_number][1])+"\n"
    #Combination dataset size:
    counts[2][0]=counts[0][0] + counts[1][0]
    counts[2][1]=counts[0][1] + counts[1][1]
    #reporting the combo data:
    print "Combination dataset: Positives: "+str(counts[2][0])+", Negatives: "+str(counts[2][1])+", TOTAL SIZE: "+str(counts[2][0]+counts[2][1])+"\n"
    print "Finished Reading."
    print "Starting to create the feature vectors for datasets..."

    # Create feature vectors
    train_vectors=[[],[],[]]
    vectorizer=[[],[],[]]
    for dataset_number in range(3):
	vectorizer[dataset_number] = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
	
	train_vectors[dataset_number] = vectorizer[dataset_number].fit_transform(train_datas[dataset_number])
    print "####################################################################"
    print "Testing comments.... (Type 'quit' to quit)"
    # Continuous user input of reviews:
    while (True):
	comment = raw_input("Please enter comment: ")
	if (comment=="quit"):
	    break
	# Perform classification with SVM, kernel=rbf
	classifier_rbf = svm.SVC(gamma='auto')

	# Perform classification with SVM, kernel=linear
	classifier_linear = svm.SVC(kernel='linear', gamma='auto')

	# Perform classification with SVM, kernel=linear
	classifier_liblinear = svm.LinearSVC()
	print "Result:"
	print ("                          %s\t%s\t%s" % (" SVMrbf "," SVMlin "," LinSVC"))

	for dataset_number in range(3):
		print datasets[dataset_number] + " Dataset syas:",
		classifier_rbf.fit(train_vectors[dataset_number], train_labels[dataset_number])
		result_SVMrbf=classifier_rbf.predict(vectorizer[dataset_number].transform([comment]))

		classifier_linear.fit(train_vectors[dataset_number], train_labels[dataset_number])
		result_SVMlinear=classifier_linear.predict(vectorizer[dataset_number].transform([comment]))

		classifier_liblinear.fit(train_vectors[dataset_number], train_labels[dataset_number])
		result_LinSVC=classifier_liblinear.predict(vectorizer[dataset_number].transform([comment]))
		
		print ("%s\t%s\t%s" % (get_polarity_name(result_SVMrbf[0]) , get_polarity_name(result_SVMlinear[0]) , get_polarity_name(result_LinSVC[0])))

	print "####################################################################"


