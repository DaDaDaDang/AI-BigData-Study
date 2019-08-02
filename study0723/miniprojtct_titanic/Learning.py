from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_and_test(model, train_data, train_label):
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=True, random_state=5)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction)*100, 2)
    print("Accuracy : ", accuracy, "%")

    log_pred = train_and_test(LogisticRegression(), train_data, train_label)

    svm_pred = train_and_test(SVC(),train_data,train_label)

    rf_pred = train_and_test(RandomForestClassifier(n_estimators=100), train_data, train_label)



