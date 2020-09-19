from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
import numpy as np

def load_iris():
    # 加载数据集，是一个字典类似Java中的map
    iris_df = datasets.load_iris()

    # 挑选出前两个维度作为x轴和y轴，你也可以选择其他维度
    x_axis = iris_df.data[:, 0]
    y_axis = iris_df.data[:, 1]
    # c指定点的颜色，当c赋值为数值时，会根据值的不同自动着色
    plt.scatter(x_axis, y_axis, c=iris_df.target)
    plt.xlabel("Sepal.Length")
    plt.ylabel("Sepal.Width")
    plt.title("Real Dataset")
    plt.show()
    return iris_df

def kmeans(n_clusters, fig=False):
    # 数据集
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target
    # kmeans模型
    model = KMeans(n_clusters=n_clusters)
    model.fit(iris_data)
    predictions = model.predict(iris_data)
    # 绘图
    if fig:
        plt.figure()
        d0 = iris_data[predictions == 0]
        plt.plot(d0[:, 0], d0[:, 1], 'r.')
        d1 = iris_data[predictions == 1]
        plt.plot(d1[:, 0], d1[:, 1], 'go')
        d2 = iris_data[predictions == 2]
        plt.plot(d2[:, 0], d2[:, 1], 'b*')
        plt.xlabel("Sepal.Length")
        plt.ylabel("Sepal.Width")
        plt.title("Kmeas Clustering")
        plt.show()
    return silhouette_score(iris_data, predictions), davies_bouldin_score(iris_data, predictions)

def AGNES(n_clusters, fig=False):
    # 数据集
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target
    # 模型
    model = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
    model.fit(iris_data)
    predictions = model.labels_
    if fig:
        plt.figure()
        d0 = iris_data[predictions == 0]
        plt.plot(d0[:, 0], d0[:, 1], 'r.')
        d1 = iris_data[predictions == 1]
        plt.plot(d1[:, 0], d1[:, 1], 'go')
        d2 = iris_data[predictions == 2]
        plt.plot(d2[:, 0], d2[:, 1], 'b*')
        plt.xlabel("Sepal.Length")
        plt.ylabel("Sepal.Width")
        plt.title("AGNES Clustering")
        plt.show()
    return silhouette_score(iris_data, predictions), davies_bouldin_score(iris_data, predictions)

def Dbscan(eps, min_samples, fig=False):
    # 数据集
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_label = iris.target
    # 模型
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(iris_data)
    predictions = model.labels_
    if fig:
        # 绘制dbscan结果
        plt.figure()
        d0 = iris_data[predictions == -1]
        plt.plot(d0[:, 0], d0[:, 1], 'r.')
        d1 = iris_data[predictions == 0]
        plt.plot(d1[:, 0], d1[:, 1], 'go')
        d2 = iris_data[predictions == 1]
        plt.plot(d2[:, 0], d2[:, 1], 'b*')
        d3 = iris_data[predictions == 2]
        plt.plot(d3[:, 0], d3[:, 1], 'k+')
        d4 = iris_data[predictions == 3]
        plt.plot(d4[:, 0], d4[:, 1], 'yx')
        plt.xlabel("Sepal.Length")
        plt.ylabel("Sepal.Width")
        plt.title("DBSCAN Clustering")
        plt.show()
    return silhouette_score(iris_data, predictions), davies_bouldin_score(iris_data, predictions)

if __name__ == "__main__":
    print(kmeans(3, True))
    print(Dbscan(0.4, 4, True))
    print(AGNES(3, True))
