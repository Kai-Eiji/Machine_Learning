import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

# The digits dataset
digits = datasets.load_digits()

#print("Digits\n", digits)

images_and_labels = list(zip(digits.images, digits.target))

#  need to flatten the image, to turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#print("Data\n",data)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# learn the digits on the first half of the digits
trainTestSplit = int(n_samples*0.5)
classifier.fit(data[:trainTestSplit], digits.target[:trainTestSplit])

# predict the value of the digit on the second half:
expected = digits.target[trainTestSplit:]
predicted = classifier.predict(data[trainTestSplit:])

#print("Classification report for classifier %s:\n%s\n"
#% (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))


# testing 
test_num = 7

plt.imshow(digits.images[test_num], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", classifier.predict(data[test_num].reshape(1,-1)))

plt.show()


