from creditcard.headfile import *

def pipline(model,x_train, y_train, x_test, y_test, disc):
    # y_test_pre = fit_predict(model, x_train, y_train, x_test, y_test)
    model.fit(x_train, y_train.ravel())  # resample 전 모델 학습
    y_test_pre = model.predict(x_test)

    cnf_matrix = confusion_matrix(y_test, y_test_pre)

    print(disc + "matrix_accuracy_score : ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100)
    print(disc + "matrix_reacll_score : ", (cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100))


