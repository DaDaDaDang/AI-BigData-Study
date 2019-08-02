# from IRIS.header_ import *
from IRIS.data_input import *

#모델정확도 확인
# logger.debug("label => ", label)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label, random_state=2019)

#모델
lr = LogisticRegression()
logger.debug(y_train)
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
# logger.debug("lr_pred => ", lr_pred)
# logger.debug(lr.predict_proba(x_test))
logger.debug("logistic regression accuracy:{:.2f}%".format(accuracy_score(y_test,lr_pred)*100))
logger.debug("logistic regression coef:{}, w:{}".format(lr.coef_, lr.intercept_))