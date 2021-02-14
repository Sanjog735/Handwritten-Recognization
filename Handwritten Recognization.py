import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv("train.csv").as_matrix()
# print(data)
clf = DecisionTreeClassifier()

# Training Data
xtrain=data[0:21000,1:]
train_label =data[0:21000,0]

clf.fit(xtrain,train_label)

# Testing Data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

d=xtest[11]
d.shape=(28,28)
plt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[11]]))
plt.show()

# (OPTIONAL) for checking Accuracy of our model
# p=clf.predict(xtest)
#
# count = 0
# for i in range(0,21000):
#     count+=1 if p[i]==actual_label[i] else 0
# print("Accuracy=",(count/21000)*100)



