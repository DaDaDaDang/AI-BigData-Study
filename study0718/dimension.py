from sklearn.decomposition import PCA
from  sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#내장 데이터 불러오기
digits = load_digits()

data = digits.data
label = digits.target
print(digits.keys())

#데이터 시각화
plt.imshow(data[0].reshape((8,8)))
plt.show()
print('Label : {}'.format(label[0]))

#차원축소
pca = PCA(n_components = 2)
pca.fit(data)
new_data = pca.transform(data)

print("원본 데이터의 차원\n{}".format(data.shape))
print("\nPCA 이후 데이터의 차원{}\n".format(new_data.shape))
plt.scatter(new_data[:, 0], new_data[:, 1], c=label, edgecolor = 'black')
plt.show()