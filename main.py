import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#We create DataFrame by accessing our dataset
#WARNING - relative path
dataset = pd.read_csv("Iris.csv")

#We want to predict values from 'Species' column
toPredict = np.array(dataset['Species'])

# Remove the data to predict from the dataset
# Axis = 1 means that we will remove a column called 'Species'
dataset = dataset.drop('Species', axis = 1)

#Preprocessing, we have 3 classes of species
i=0
for spicie in toPredict:
    if spicie == "Iris-setosa":
        toPredict[i] = 0
    elif spicie == "Iris-versicolor":
        toPredict[i] = 1
    else:
        toPredict[i] = 2
    i+=1

#Store column labels
metrics_list = list(dataset.columns)

#Store metrics
metrics = np.array(dataset)

#Divide dataset into training and testing set, we can change random_state to change the shuffling of the data
train_metrics, test_metrics, train_toPredict, test_toPredict = train_test_split(metrics, toPredict, test_size = tSize, random_state = intForRandomState)
