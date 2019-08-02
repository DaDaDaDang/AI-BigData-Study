from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from data import boston as bd
from sklearn.metrics import r2_score
from util.logfile import logger
import  matplotlib.pyplot as plt
import numpy as np
# logger = logging.getLogger(__name__)


#단순 백터머신회귀
svm_regr = SVR(gamma='scale')
# svm_regr = SVR(C=1.0, kernel='rbf',gamma='scale')

svm_regr.fit(bd.x_train['RM'].values.reshape((-1,1)),bd.y_train)

y_pred = svm_regr.predict(bd.x_test['RM'].values.reshape((-1,1)))

print('단순 서포트 벡터 머신 회귀, R2 : {:.4f}'.format(r2_score(bd.y_test, y_pred)))

#전체 변수 벡터 머신 회귀
# svm_regr = SVR(gamma='scale')

# svm_regr.fit(bd.x_train,bd.y_train)

# y_pred = svm_regr.predict(bd.x_test)

logger.debug('svm 전체 회귀 {:.4f}'.format(r2_score(bd.y_test, y_pred)))
plt.scatter(bd.x_test['RM'], bd.y_test, s=10, c='black')

line_x = np.linspace(np.min(bd.x_test['RM']),np.max(bd.x_test['RM']),100)
line_y = svm_regr.predict(line_x.reshape(-1,1))

plt.plot(line_x, line_y, c='red')
plt.legend(['SVM line', 'x_test'], loc = 'upper left')
plt.show()
# print('svm 전체 회귀 {:.4f}'.format(r2_score(bd.y_test, y_pred)))