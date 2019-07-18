# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:21:51 2019

@author: 123
"""

## Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

## Importing the dataset
df = pd.read_csv('spam.tsv', delimiter = '\t',quoting = 3)
df = df[['v1','v2']]
df.rename(columns = {'v1':'Result','v2' : 'Message'}, inplace = True)
#df[df['Result'] == 'ham"""']  This is wierd result, hence we are skipping these two rows.
df = df[df['Result'] != 'ham"""']

## Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,5574):
    if i == 99 or i == 2793:
        continue
    review = re.sub('[^a-zA-Z]',' ',df['Message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word ) for word in review if not word in 
              set(stopwords.words('English'))]
    review = ' '.join(review)
    corpus.append(review)
    
## Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,0].values

## Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## Splitting into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

## Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(cv, open('cv.pkl','wb'))

classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
## Predicting Test set results
y_pred = classifier.predict(X_test)
comment = 'Hey! How are you ?'
comment = re.sub('[^a-zA-Z]',' ',comment)
comment = comment.lower()
comment = comment.split()
comment = [ps.stem(word ) for word in comment if not word in 
              set(stopwords.words('English'))]
comment = ' '.join(comment)
comment = [comment]
vec = cv.transform(comment).toarray()
singlePred = classifier.predict(vec)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)