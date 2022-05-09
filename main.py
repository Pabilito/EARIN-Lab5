import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import timeit
import csv
from sklearn.metrics import plot_confusion_matrix, mean_absolute_error
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
hidden_layers = [   
                    (100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                ]
times = []
accuracies = []
mse = []
mae = []
iter = 0
iter2 = 0

for layers in hidden_layers:
    start = timeit.default_timer()
    #Solver is stochastic gradient descent
    #Also I added more iteration to assure convergence
    clf = MLPClassifier(activation = "tanh", solver="sgd", hidden_layer_sizes=layers, random_state = 1, max_iter = 10000).fit(train_metrics, train_toPredict)
    stop = timeit.default_timer()
    times.append("{0:0.3f}".format(stop - start))
    accuracies.append(clf.score(test_metrics, test_toPredict))
    proba = clf.predict_proba(test_metrics)
    mse_temp = 0
    mae_temp = 0
    for flower in test_toPredict:
        if flower != "Iris-setosa":
            mse_temp = mse_temp + proba[iter2][0] * proba[iter2][0]
            mae_temp = mae_temp + proba[iter2][0]
        if flower != "Iris-versicolor":
            mse_temp = mse_temp + proba[iter2][1] * proba[iter2][1]
            mae_temp = mae_temp + proba[iter2][1]
        if flower != "Iris-virginica":
            mse_temp = mse_temp + proba[iter2][2] * proba[iter2][2]
            mae_temp = mae_temp + proba[iter2][2]
        iter2 = iter2 + 1
    iter = iter + 1
    mse.append(mse_temp/iter2)
    mae.append(mae_temp/iter2)
    iter2 = 0

iter = 0
#Save results to file
filename = "TestResults/MLP2.csv"
with open(filename,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=';')
    csvWriter.writerow(["Hidden layer sizes","Train time","Accuracy", "MSE", "MAE"])
    for _ in times:
        csvWriter.writerow([hidden_layers[iter], times[iter], accuracies[iter], mse[iter], mae[iter]])
        iter = iter + 1

fig = plot_confusion_matrix(clf, test_metrics, test_toPredict, display_labels=["Iris-Setosa","Iris-Versicolor","Iris-Virginica"])
fig.figure_.suptitle("Confusion Matrix")
plt.show()

#Two cost functions to be done
#1. Cross entropy 
#https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451