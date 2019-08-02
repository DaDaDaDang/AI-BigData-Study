import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import  RandomForestClassifier
from util.logfile import logger
import matplotlib.pyplot as plt

iris = load_iris()