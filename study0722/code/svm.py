import matplotlib.pyplot as plt
import numpy as np

#import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

#load the digits dataset
digits = datasets.load_digits()
print('Digits dataset keys \n{}'.format(digits.keys()))

print('dataset target name : \n{}'.format(digits.target_names))
print('shape of dataset : {} \n and target:{}'.format(digits.data.shape, digits.target.shape))
print('shape of the images : {}'.format(digits.images.shape))

