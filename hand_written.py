from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import sys

def main():
    (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
    for k in range(1, 20):
        error_rate=0
        knn = KNeighborsClassifier(n_neighbors=k)
        print("fitting...")
        knn.fit(Xtrain, ytrain)
        print("done.")
        for i in range(Xtest.shape[0]):
            p=knn.predict(Xtest[i])
            if p != ytest[i]:
                error_rate += 1/Xtest.shape[0]
            sys.stdout.write("\r{}%           ".format(100*error_rate))
        print("error_rate({0}) is {1}".format(k, error_rate))

if __name__=='__main__':
    main()