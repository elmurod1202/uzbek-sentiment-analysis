import os
import re

import warnings
warnings.simplefilter("ignore", UserWarning)

# Importing libraries For machine learning based analysis:
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np 
from string import punctuation

from tqdm import tqdm


from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

import scipy
from scipy.sparse import hstack

# Importing libraries For deep learning based analysis
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model



# name of the folder that input data contains:
data_dir = "data"
# names of classes to analyse:
classes = ['pos', 'neg']
# names of the input files for each class:
# 1: Annotated dataset:
# filenames = ["UZ_positive.txt" , "UZ_negative.txt"]

# 2: Translated dataset:
filenames = ["positive10kUZ.txt" , "negative10kUZ.txt"]


# Initial variables to be used:
train_data = []
train_labels = []

# Read the data
sum_count_train=0;
for polarity in range(2):
    print ("Polarity class: "+classes[polarity]+ ", Opening file: "+filenames[polarity]+":\n")
    with open(os.path.join(data_dir, filenames[polarity]), 'r') as infile:
        count=0
        for line in infile.readlines():
            count+=1
            review=line.strip('\n')
            train_data.append(review)
            label=0
            if (classes[polarity]=='pos'):
                label=1
            train_labels.append(label)
        print ("\tNumber of reveiws: "+str(count))
        sum_count_train+=count
#reporting the read data:
print ("Total number of reveiws: "+str(sum_count_train))

#.to_csv('./data/cleaned_text.csv')

# --- build the model

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

print (len(x_train), len(x_test), len(y_train), len(y_test))

#Saving the test labels for future use
pd.DataFrame(y_test).to_csv('./predictions/y_true.csv', index=False, encoding='utf-8')





# 4 - Recurrent Neural Network without pre-trained embedding

print ("### 4 - Recurrent Neural Network without pre-trained embedding")
MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(train_data)
#print (x_train[0])
#print (tokenizer.texts_to_sequences([x_train[0]]))
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)


MAX_LENGTH = 45
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)




# 6 - Multi-channel Convolutional Neural Network


print ("6 - Multi-channel Convolutional Neural Network\n")


def get_cnn_model():
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))
    
    filter_sizes = [2, 3, 5]
    num_filters = 256
    drop = 0.3

    inputs = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedding = Embedding(input_dim=MAX_NB_WORDS,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=MAX_LENGTH,
                                trainable=True)(inputs)

    reshape = Reshape((MAX_LENGTH, embedding_dim, 1))(embedding)
    conv_0 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[0], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)

    conv_1 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[1], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, 
                    kernel_size=(filter_sizes[2], embedding_dim), 
                    padding='valid', kernel_initializer='normal', 
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[0] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_0)

    maxpool_1 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[1] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_1)

    maxpool_2 = MaxPool2D(pool_size=(MAX_LENGTH - filter_sizes[2] + 1, 1), 
                          strides=(1,1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
cnn_model_multi_channel = get_cnn_model()

filepath="./models/cnn_multi_channel/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 20

history = cnn_model_multi_channel.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)
