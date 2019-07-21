from sklearn.decomposition import PCA
from keras.datasets import mnist
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier


def main():
    # データをロード
    (train_img, train_l), (test_img, test_l) = mnist.load_data()

    print("train_img:{0}, test_img:{1}".format(train_img.shape, test_img.shape))
    # 教師データとテストデータを結合
    data = np.vstack((train_img, test_img))
    reshaped_img = []

    # 28x28の2次元画像データを1次元にする
    print("data reshaping...")
    for i in range(data.shape[0]):
        reshaped_img.append(np.ravel(data[i]))
    print("done.")

    # データに対し、次元数30で主成分分析を行う
    print("pca step running...")
    pca = PCA(n_components=30)
    pca.fit(reshaped_img)
    print("done. pca components' shape:", pca.components_.shape)

    # TODO
    # データを教師用とテスト用に分割
    #train_components = pca.components_[0:train_img.shape[0]]
    #test_components = pca.components_[-test_img.shape[0]:]
    #print("train:{0}, test:{1}".format(len(train_components), len(test_components)))

    # TODO
    # 次元数を減らしたデータに対し、knnを実行
    #knn = KNeighborsClassifier(n_neighbors=3)
    #print("fitting...")
    #knn.fit()
    #print("done.")

if __name__=='__main__':
    main()