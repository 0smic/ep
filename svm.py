import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

X = iris.data[:, [2, 3]]

y = (iris.target != 0).astype(int)

model = SVC(kernel='linear', C=1)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

y_boundary = -(w[0] * x + b) / w[1]
y_margin1 = -(w[0] * x + b - 1) / w[1]
y_margin2 = -(w[0] * x + b + 1) / w[1]

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='blue',
    label='Setosa'
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='red',
    label='Non-Setosa'
)

plt.plot(x, y_boundary, 'k-', label='Decision Boundary')


plt.plot(x, y_margin1, 'k--', label='Margin')

plt.plot(x, y_margin2, 'k--')

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='green',
    label='Support Vectors'
)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear SVM - Setosa vs Non-Setosa')
plt.legend()
plt.show()

import seaborn as sns

data = sns.load_dataset('iris')
print(data.head())
