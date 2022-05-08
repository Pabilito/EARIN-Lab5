import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import timeit
import csv
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#We create DataFrame by accessing our dataset
#WARNING - relative path
dataset = pd.read_csv("Iris.csv")

#We want to predict values from 'Species' column
toPredict = np.array(dataset['Species'])

# Remove the data to predict from the dataset
# We will just focus on columns 3 and 4 with petal width and length
dataset = dataset.iloc[0:150, 2:4]

#Store column labels
metrics_list = list(dataset.columns)

#Store metrics
metrics = np.array(dataset)

#Divide dataset into training and testing set, we can change random_state to change the shuffling of the data
train_metrics, test_metrics, train_toPredict, test_toPredict = train_test_split(metrics, toPredict, test_size = 0.75, random_state = 42)

#Yes, I know, I could make a loop
#But instead I almost made a flag of Nepal
hidden_layers = [   (10),
                    (10, 10),
                    (10, 10, 10),
                    (10, 10, 10, 10),
                    (10, 10, 10, 10, 10),
                    (10, 10, 10, 10, 10, 10),
                    (10, 10, 10, 10, 10, 10, 10),
                    (10, 10, 10, 10, 10, 10, 10, 10),
                    (10, 10, 10, 10, 10, 10, 10, 10, 10),
                    (10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
                    (25),
                    (25, 25),
                    (25, 25, 25),
                    (25, 25, 25, 25),
                    (25, 25, 25, 25, 25),
                    (25, 25, 25, 25, 25, 25),
                    (25, 25, 25, 25, 25, 25, 25),
                    (25, 25, 25, 25, 25, 25, 25, 25),
                    (25, 25, 25, 25, 25, 25, 25, 25, 25),
                    (25, 25, 25, 25, 25, 25, 25, 25, 25, 25),
                    (100),
                    (100, 100),
                    (100, 100, 100),
                    (100, 100, 100, 100),
                    (100, 100, 100, 100, 100),
                    (100, 100, 100, 100, 100, 100),
                    (100, 100, 100, 100, 100, 100, 100),
                    (100, 100, 100, 100, 100, 100, 100, 100),
                    (100, 100, 100, 100, 100, 100, 100, 100, 100),
                    (100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                ]
times = []
accuracies = []
iter = 0

for layers in hidden_layers:
    start = timeit.default_timer()
    #Solver is stochastic gradient descent
    #Also I added more iteration to assure convergence
    clf = MLPClassifier(activation = "tanh", solver="sgd", hidden_layer_sizes=layers, random_state = 1, max_iter = 10000).fit(train_metrics, train_toPredict)
    stop = timeit.default_timer()
    times.append("{0:0.3f}".format(stop - start))
    accuracies.append(clf.score(test_metrics, test_toPredict))
    iter = iter + 1

iter = 0
filename = "TestResults/MLP.csv"
with open(filename,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=';')
    csvWriter.writerow(["Hidden layer sizes","Train time","Accuracy"])
    for _ in times:
        csvWriter.writerow([hidden_layers[iter], times[iter], accuracies[iter]])
        iter = iter + 1

fig=plot_confusion_matrix(clf, test_metrics, test_toPredict, display_labels=["Setosa","Versicolor","Virginica"])
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()