import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sample = np.random.randint(low=1, high=100, size=10)
print('Original sample: %s' % sample)
print('Sample mean: %s' % sample.mean())

resamples = [np.random.choice(sample, size=sample.shape) for i in range(100)]
print('Number of bootstrap re-samples: %s' % len(resamples))
print('Example re-sample: %s' % resamples[0])

resamplesMean = np.array([resample.mean() for resample in resamples])
print(resamplesMean)
print('Mean of re-samples\' means: %s' % resamplesMean.mean())

# Let's train a random forest with scikit-learn:
X, y = make_classification(n_samples=1000,
                           n_features=100,
                           n_informative=20,
                           n_clusters_per_class=2,
                           random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

clf = RandomForestClassifier(n_estimators=10, random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# We create a dataset with 1,000 instances. Of the 100 features, 20 are informative;
# the remainder are redundant combinations of the information features, or noise. We then
# train and evaluate a single decision tree, followed by a random forest with 10 trees.
# The random forest's F1 precision, recall, and F1 scores are greater.
