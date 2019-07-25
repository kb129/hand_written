from sklearn.decomposition import PCA
from keras.datasets import mnist
import numpy as np
import sys
import time
from sklearn.neighbors import KNeighborsClassifier

# データをロード
(train_img, train_l), (test_img, test_l) = mnist.load_data()

print("train_img:{0}, test_img:{1}".format(train_img.shape, test_img.shape))
# 教師データとテストデータを結合
data = np.vstack((train_img, test_img))
   
reshaped_img = np.zeros((70000, 28 * 28))

# 28x28の2次元画像データを1次元にする
print("data reshaping...")
for i in range(data.shape[0]):
    reshaped_img[i] = np.ravel(data[i])
print("done.")

# ---------- function define ----------

def pca_knn_main(dim, k):
    # 実行時間計測開始
    start_time = time.time()
    
    # データに対し、次元数dimで主成分分析を行う
    print("pca step running...")
    pca = PCA(n_components=dim)
    pca.fit(reshaped_img)
    components = pca.components_
    print("done. pca components' shape:", components.shape)

    # 次元数dimで、それぞれの主成分に射影
    print("projecting...")
    projection = pca.transform(reshaped_img)
    print("done. projection.shape:", projection.shape)

    # データを教師用とテスト用に分割
    train_data = projection[0:train_img.shape[0]]
    test_data = projection[-test_img.shape[0]:]
    print("train:{0}, test:{1}".format(train_data.shape, test_data.shape))

    # 次元数を減らしたデータに対し、knnを実行
    knn = KNeighborsClassifier(n_neighbors=k)
    print("fitting...")
    knn.fit(train_data, train_l)
    print("done.")

    # ラベルを予想
    print("predicting...")
    predicted_label = knn.predict(test_data)
    print("done.")
    print("error rate is now calculating...")

    # 誤差率を計算
    error_rate = 0
    for i in range(test_data.shape[0]):
        if predicted_label[i] != test_l[i]:
            error_rate += 1
    print("done.")
    rate = error_rate / test_l.shape[0]
    print("error rate is ", 100 * rate, "%")
    # 計測終了
    end_time = time.time()

    # 誤差率と実行時間を返す
    return rate, (end_time - start_time)

# ---------- end of pca_knn_main ----------

k = 3
# 実行する次元数
#check_dims = [10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 
check_dims = [100, 200, 500]
with open("time.csv", "w") as f:
    f.write("dim, error_rate, process_time\n")
    for dim in check_dims:
        rate, p_time = pca_knn_main(dim, k)
        outstr = "{}, {:f}, {}\n".format(dim, rate, p_time)
        print(outstr)
        f.write(outstr)

