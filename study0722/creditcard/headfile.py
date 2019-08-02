from data import creditcard
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score
from sklearn.svm import SVC
#x_train, x_test, y_train, y_test, x_train_res, y_trian_res
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")