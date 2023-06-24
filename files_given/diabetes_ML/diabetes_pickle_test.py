import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
import pickle 

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
knn = KNeighborsClassifier(n_neighbors = 47)
knn.fit(Xtrain,ytrain)
training_accuracy.append(knn.score(Xtrain,ytrain))
test_accuracy.append(knn.score(Xtest,ytest))


print("Accuracy of the training set for 47 NN: {}".format(training_accuracy))
print("Accuracy of the test set for 47 NN: {}".format(test_accuracy))


knn_pickle = open("knn_pickle_file", "wb")
pickle.dump(knn, knn_pickle)
knn_pickle.close()

loaded_model = pickle.load(open("knn_pickle_file", "rb"))
result = loaded_model.predict(Xtest)
print(knn.score(Xtest, ytest))

