from sklearn.tree import DecisionTreeRegressor
from data import boston as bd
from sklearn.metrics import r2_score
import  matplotlib.pyplot as plt
# from mglearn.plots import plot_animal_tree
import numpy as np 

dt_regr = DecisionTreeRegressor(max_depth=4, random_state=2019)

dt_regr.fit(bd.x_train['RM'].values.reshape((-1,1)), bd.y_train)
y_pred = dt_regr.predict(bd.x_test['RM'].values.reshape((-1,1)))

print('단순 결정 트리 회귀, R2 : {:.4f}'.format(r2_score(bd.y_test, y_pred)))

# plot_animal_tree()
# plt.show()

#10개만
line_x=np.linspace(np.min(bd.x_test['RM']),np.max(bd.x_test['RM']),10)
line_y=dt_regr.predict(line_x.reshape(-1,1))

#10개만 선별해서 그림
plt.scatter(bd.x_test['RM'].values.reshape(-1,1),bd.y_test,s=10,c='black')
plt.plot(line_x, line_y,c='red')
plt.legend(['DT Regression line', 'x_test'], loc = 'upper left')
plt.show()

#13개 변수 사용 test
#학습
dt_regr.fit(bd.x_train, bd.y_train)
y_pred_all = dt_regr.predict(bd.x_test)
print("단순 결정 트리 (all변수) 회귀:{:.4f}".format(r2_score(bd.y_test,y_pred_all)))

import graphviz
from sklearn.tree import export_graphviz
export_graphviz(dt_regr, out_file='boston.dot', class_names=bd.label, feature_names=bd.columns, filled=True, impurity=False)
with open('boston.dot') as file_reader :
    dot_graph = file_reader.read()
dot = graphviz.Source(dot_graph)
dot.render(filename='boston.png')