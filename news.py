from sklearn import svm 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB 
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
import pandas as pd
df = pd.read_csv('news-train.csv')
X = df["HEADER"]
y = df["CAT"]
vect = TfidfVectorizer(analyzer='word',stop_words='english')
X = vect.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=5)
C = 1.0
clf=svm.SVC(kernel='linear', C=C)
clf = svm.LinearSVC(C=C, dual = True, loss = 'hinge')
fit=clf.fit(X_train, y_train)
print("train_score:", fit.score(X_train, y_train))
print("val_score:", fit.score(X_test, y_test))
df_test = pd.read_csv('test-full.csv')
x_val = vect.transform(df_test ["HEADER"].values)
y_pred =clf.predict(x_val)
print(y_pred)
open("answer.txt", "w").write("\n".join(y_pred))