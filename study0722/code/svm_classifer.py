from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_samples = len(digits.images)
data_images = digits.images.reshape((n_samples, -1))

x_train, x_test, y_train, y_test = train_test_split(data_images, digits.target)
print('Training data and target sizes : \n{}, {}'.format(x_train.shape, y_train.shape))
print('Test data and target sizes : \n{}, {}'.format(x_test.shape, y_test.shape))

classifier = svm.SVC(gamma=0.001)
#fit to the train in data
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print("Classification report for classifier %s:\n%s\n"%(classifier,metrics.classification_report(y_test, y_pred)))
