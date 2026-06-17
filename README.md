from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

print("First five records:")
print(iris_df.head())

print("
Number of rows and columns:")
print(iris_df.shape)

print("
Column names:")
print(iris_df.columns)

print("
Missing values:")
print(iris_df.isnull().sum())

print("
Summary statistics:")
print(iris_df.describe())

print("
Number of samples in each class:")
print(iris_df["target"].value_counts())
