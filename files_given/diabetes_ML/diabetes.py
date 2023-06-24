import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")
df.head()

train, test = np.split(df.sample(frac=1), [int(.6*len(df))])

def scale_dataset(dataframe, oversample=False):
  #Fancy way of saying that X is all the features, and y is our outcome
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y


train, Xtrain, ytrain = scale_dataset(train, oversample=True)
test, Xtest, ytest = scale_dataset(test, oversample=False)

training_accuracy = []
test_accuracy = []


#KNN for different k nearest neighbor from 1-30
neighbors_setting = range(1,50)
for n_neighbors in neighbors_setting:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(Xtrain,ytrain)
    training_accuracy.append(knn.score(Xtrain,ytrain))
    test_accuracy.append(knn.score(Xtest,ytest))


plt.plot(neighbors_setting,training_accuracy, label="Accuracy of training dataset")
plt.plot(neighbors_setting, test_accuracy, label="Accuracy of testing dataset")
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbors")
plt.legend()
plt.show()

print("Accuracy of the training set for 12 NN: {:3f}".format(training_accuracy[13]))
print("Accuracy of the test set for 12 NN: {:3f}".format(test_accuracy[13]))


log_reg = LogisticRegression()
log_reg.fit(Xtrain, ytrain)

print('Accuracy on the training set: {:.4f}'.format(log_reg.score(Xtrain,ytrain)))
print('Accuracy on the testing set: {:.3f}'.format(log_reg.score(Xtest,ytest)))