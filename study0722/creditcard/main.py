from creditcard.smoteExample import *

if __name__ == "__main__" :
    x_train = creditcard.x_train
    y_train = creditcard.y_train
    x_test = creditcard.x_test
    y_test = creditcard.y_test

    x_smote = creditcard.x_train_res
    y_smote = creditcard.y_train_res

    # logisticR(x_train, y_train, x_test, y_test, "smote전 + logisticR") #smote전
    # logisticR(x_smote, y_smote, x_test, y_test, "smote후 + logisticR") #smote후
    # rf(x_train, y_train, x_test, y_test, "smote전 + rf")
    # rf(x_train, y_train, x_test, y_test, "smote전 + rf")

    lr = LogisticRegression()
    rf = RandomForestClassifier()
    svc = SVC()

    pipline(lr, x_train, y_train, x_test, y_test, "smote전 + logisticR")
    pipline(lr, x_smote, y_smote, x_test, y_test, "smote후 + logisticR")
    pipline(rf, x_train, y_train, x_test, y_test, "smote전 + rf")
    pipline(rf, x_smote, y_smote, x_test, y_test, "smote후 + rf")
    pipline(svc, x_train, y_train, x_test, y_test, "smote전 + svc")
    pipline(svc, x_smote, y_smote, x_test, y_test, "smote후 + svc")
