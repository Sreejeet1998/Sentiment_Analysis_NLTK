import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

#IMPORTIND DATASETS
dataset = pd.read_csv("C:\\Users\\Sreejeet\\Downloads\\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
print(dataset)

#CLEANING TEXTS simplyfying deep clean the texts
nltk.download('stopwords')

corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

#CREATING BAG OF WORDS
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,-1].values

print(len(X[0]))

#SPLITTING THE DATASET INTO TRAINIG SET & TEST SET
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

#TRAINING THE NAIVE BAYES MODEL ON TRAINIG SET
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#PREDICTING THE TEST SET RESULTS
Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#MAKING THE CONFUSION METRIX
cm = confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix =\n",cm)
print("ACCURACY FOR NAIVE BAYES =",accuracy_score(Y_test,Y_pred))

#TRAINING THE KNN MODEL ON TRAINIG SET
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))