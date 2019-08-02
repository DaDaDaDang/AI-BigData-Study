import matplotlib.pyplot as plt
# import numpy as np

#import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

#the images are also included in the dataset as digits.images
for i in range(0,4) :
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.imshow(digits.images[i])
    plt.title('Training:{}'.format(digits.target[i]))
plt.show()