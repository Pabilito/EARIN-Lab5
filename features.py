
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#Show on the matrix graph how metrics are distributed
dataset = pd.read_csv('Iris.csv')
scatter_matrix(dataset, alpha=1, figsize=(12, 12))
plt.show()

#SKLEARN allows us to fetch this dataset and also gives description of its properties
#It is an alternative for reading from file
from sklearn.datasets import load_iris
dataset = load_iris()
print(dataset['DESCR'])