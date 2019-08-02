# from IRIS.header_ import *
# from IRIS.data_input import *
# from IRIS.model import *
from IRIS.svc_data import *

#데이터 프레임 만들기
df = pd.DataFrame(dt.feature_importances_.reshape((1,-1)),columns=columns, index=['feature_importances_'])
logger.debug(df.columns) #petal length (cm) 가장 중요하다
######################################rf

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
logger.debug("rf accuracy:{:.2f}%".format(accuracy_score(y_test, rf_pred)*100))
df = pd.DataFrame(rf.feature_importances_.reshape((1,-1)), columns=columns, index=['feature_importances_'])
logger.debug(rf.score(x_test,y_test)) #어떤 score를 쓰는지 확인이 어렵다.