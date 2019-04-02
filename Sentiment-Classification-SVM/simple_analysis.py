import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

# name of the folder that input data contains:
data_dir = "data"
# names of classes to analyse:
classes = ['pos', 'neg']
# names of the input files for each class:
filenames = ["UZ_positive.txt" , "UZ_negative.txt"]

# Initial variables to be used:
train_data = []
train_labels = []

# Read the data
sum_count_train=0;
for polarity in range(2):
    print "Polarity class: "+classes[polarity]+ ", Opening file: "+filenames[polarity]+":\n"
    with open(os.path.join(data_dir, filenames[polarity]), 'r') as infile:
	count=0
	for line in infile.readlines():
	    count+=1
	    review=line.strip('\n')
	    train_data.append(review)
	    train_labels.append(classes[polarity])
	print "\tNumber of reveiws: "+str(count)
	sum_count_train+=count
#reporting the read data:
print "Total number of reveiws: "+str(sum_count_train)



# --- build the model

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)

# -
count = CountVectorizer()
temp = count.fit_transform(X_train)

tdif = TfidfTransformer()
temp2 = tdif.fit_transform(temp)

text_regression = LogisticRegression()
model = text_regression.fit(temp2, y_train)

prediction_data = tdif.transform(count.transform(X_test))

predicted = model.predict(prediction_data)

# --- make predictions

print(np.mean(predicted == y_test))

# --- have some fun with the model

print(model.predict(tdif.transform(count.transform(["Bu ilova juda yaxshi ekan, menga yoqdi"]))))
