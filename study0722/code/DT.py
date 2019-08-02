import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from data import wine as w
import graphviz
from sklearn.tree import export_graphviz
from util.logfile import logger

#pre-pruning의 방법 중 하나는 깊이의 최대를 설정
tree = DecisionTreeClassifier(random_state=0)
tree.fit(w.x_train, w.y_train)
score_tr = tree.score(w.x_train, w.y_train)
score_te = tree.score(w.x_test, w.y_test)
print('DT훈련 세트 정확도{:.3f}'.format(score_tr))
print('DT테스트 세트 정확도{:.3f}'.format(score_te))

tree1 = DecisionTreeClassifier(max_depth=2, random_state=0)
tree1.fit(w.x_train, w.y_train)
score_tr1 = tree1.score(w.x_train,w.y_train)
score_te1 = tree1.score(w.x_test, w.y_test)
print('DT훈련 세트 정확도{:.3f}'.format(score_tr1))
print('DT테스트 세트 정확도{:.3f}'.format(score_te1))

#tree modul의 export graphviz함수를 이용해 tree를 시각화 -> pdf로
export_graphviz(tree1, out_file='tree1.dot', class_names=w.classname, feature_names=w.columns, filled=True, impurity=False)
with open('tree1.dot') as file_reader :
    dot_graph = file_reader.read()
dot = graphviz.Source(dot_graph)
dot.render(filename='tree1.png')

#tree modul의 export graphviz함수를 이용해 tree를 시각화 -> 그래프
print('특성 중요도 첫번째 :\n{}'.format(tree.feature_importances_))
print('특성 중요도 두번째 :\n{}'.format(tree1.feature_importances_))
logger.debug('특성 중요도 첫번째 :\n{}'.format(tree.feature_importances_))
logger.debug('특성 중요도 두번째 :\n{}'.format(tree1.feature_importances_))


print("wine.data.shape=>",w.wine.data.shape)
n_feature = w.wine.data.shape[1]
print(n_feature)
idx = np.arange(n_feature)
print("idx => ", idx)
feature_imp = tree.feature_importances_
plt.barh(idx, feature_imp, align='center')
plt.yticks(idx, w.columns)
plt.xlabel('feature importance', size = 15)
plt.ylabel('feature', size = 15)
   
plt.show()
