# from IRIS.header_ import *
# from IRIS.data_input import *
from IRIS.model import *

#커널 linear(선형), ploy(다항식), RBF(방사기저), Hyper-tangent(쌍곡선 탄젠트 함수)
svc=SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
logger.debug("svm accuracy:{:.2f}%".format(accuracy_score(y_test, y_pred)*100))
###############dt###################
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
dt_y_pred = dt.predict(x_test)
logger.debug("dt accuracy:{:.2f}%".format(accuracy_score(y_test, dt_y_pred)*100))
logger.debug(dt.feature_importances_)#중요도보기

