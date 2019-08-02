from creditcard.headfile import *

# def fit_predict(data, x_train, y_train, x_test, y_test) :
#     data.fit(x_train, y_train.ravel()) #resample 전 모델 학습
#     y_test_pre = data.predict(x_test)
#     return y_test_pre

def pipline(model,x_train, y_train, x_test, y_test, disc):
    # y_test_pre = fit_predict(model, x_train, y_train, x_test, y_test)
    model.fit(x_train, y_train.ravel())  # resample 전 모델 학습
    y_test_pre = model.predict(x_test)

    # print(disc + "accuracy_score    :{:.2f}%".format(accuracy_score(y_test, y_test_pre) * 100))
    # print(disc + "recall_score      :{:.2f}%".format(recall_score(y_test, y_test_pre) * 100))
    # print(disc + "precision_score   :{:.2f}%".format(precision_score(y_test, y_test_pre) * 100))
    # print(disc + "roc_auc_score     :{:.2f}%".format(roc_auc_score(y_test, y_test_pre) * 100))

    cnf_matrix = confusion_matrix(y_test, y_test_pre)
    # print(disc + "===>\n", cnf_matrix)  # matrix count
    # print("cnf_matrix_test[0, 0] >=", cnf_matrix[0, 0])
    # print("cnf_matrix_test[0, 1] >=", cnf_matrix[0, 1])
    # print("cnf_matrix_test[1, 0] >=", cnf_matrix[1, 0])
    # print("cnf_matrix_test[1, 1] >=", cnf_matrix[1, 1])

    print(disc + "matrix_accuracy_score : ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100)
    print(disc + "matrix_reacll_score : ", (cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100))

# def logisticR(x_train, y_train, x_test, y_test, disc) :
#     lr = LogisticRegression()
#     y_test_pre = fit_predict(lr, x_train, y_train, x_test, y_test)
#
#     print(disc + "accuracy_score    :{:.2f}%".format(accuracy_score(y_test, y_test_pre)*100))
#     print(disc + "recall_score      :{:.2f}%".format(recall_score(y_test, y_test_pre)*100))
#     print(disc + "precision_score   :{:.2f}%".format(precision_score(y_test, y_test_pre)*100))
#     print(disc + "roc_auc_score     :{:.2f}%".format(roc_auc_score(y_test, y_test_pre)*100))
#
#     cnf_matrix = confusion_matrix(y_test, y_test_pre)
#     print(disc + "===>\n", cnf_matrix) #matrix count
#     print("cnf_matrix_test[0, 0] >=", cnf_matrix[0,0])
#     print("cnf_matrix_test[0, 1] >=", cnf_matrix[0,1])
#     print("cnf_matrix_test[1, 0] >=", cnf_matrix[1,0])
#     print("cnf_matrix_test[1, 1] >=", cnf_matrix[1,1])
#
#     print(disc + "matrix_accuracy_score : ", (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[0,1]+cnf_matrix[1,0]+cnf_matrix[1,1])*100)
#     print(disc + "matrix_reacll_score : ", (cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])*100))

# def rf(x_train, y_train, x_test, y_test, disc) :
#     rf = RandomForestClassifier()
#     y_test_pre = fit_predict(rf, x_train, y_train, x_test, y_test)
#
#     cnf_matrix = confusion_matrix(y_test, y_test_pre)
#     print(disc + "matrix_accuracy_score : ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
#                 cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100)
#     print(disc + "matrix_reacll_score : ", (cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100))
