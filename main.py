import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#We create DataFrame by accessing our dataset
#WARNING - relative path
dataset = pd.read_csv("Iris.csv")

#We want to predict values from 'Species' column
toPredict = np.array(dataset['Species'])

# Remove the data to predict from the dataset
# We will just focus on columns 3 and 4 with petal width and length
dataset = dataset.iloc[0:150, 2:4]

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
train_metrics, test_metrics, train_toPredict, test_toPredict = train_test_split(metrics, toPredict, test_size = 0.75, random_state = 42)

#Activation functions
def activation_sigmoid(x):
    return 1/(1 + np.exp(-x))
def activation_tanh(x):
    return np.tanh(x)

#Variables used during training
inputLayerNeurons = 2                                              # Input layer neurons
hiddenLayersNeurons = [3,3]                                        # Hidden layer neurons
outputLayerNeurons = 2                                             # Output layer neurons
learningRate = 0.01                                                # Learning rate
epochs = 100                                                       # Epochs
iasHiddenValue = np.random.uniform(-1, 1)                          # Bias for hidden Layer
BiasOutputValue = np.random.uniform(-1, 1)                         # Bias for output Layer

