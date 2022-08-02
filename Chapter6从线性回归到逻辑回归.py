import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('sms.csv')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'], random_state=11)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Accuracies: %s' % scores)
print('Mean accuracy: %s' % np.mean(scores))

precisions = cross_val_score(classifier, X_train, y_train,
                             cv=5,
                             scoring='precision')
print('Precision: %s' % np.mean(precisions))

recalls = cross_val_score(classifier, X_train, y_train,
                          cv=5,
                          scoring='recall')
print('Recall: %s' % np.mean(recalls))
