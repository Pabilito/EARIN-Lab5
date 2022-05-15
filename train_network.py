from torch.nn import Module, Tanh, CrossEntropyLoss, Sequential, Flatten, Linear
from torch.optim import SGD
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MyCoolDataset(Dataset):
    def __init__(self, x_train, x_test, y_train, y_test, train=True):
        if train:
            self.x_data, self.y_data = x_train, y_train
        else:
            self.x_data, self.y_data = x_test, y_test

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]


class MLPModel(Module):
    def __init__(self, input_size, hidden_size):
        super(MLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = Linear(input_size, hidden_size)
        self.tanh = Tanh()
        self.fc2 = Linear(hidden_size, 1)
        '''
        self.layers = Sequential(
            #            Flatten(),
            Linear(32 * 32 * 3, 64),
            #Linear(size_of_each_input_sample, size_of_each_output_sample),
            #            Linear(size_of_each_input_sample, size_of_each_output_sample),
            Tanh(),
            Linear(64, 32),
            #Linear(size_of_each_input_sample, size_of_each_output_sample),
            Tanh())
        '''

    def forward(self, x):
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        output = self.tanh(output)
        return output


def main():
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
    train_metrics, test_metrics, train_toPredict, test_toPredict = train_test_split(
        metrics, toPredict, test_size=0.25, random_state=42)
    train_metrics = torch.from_numpy(train_metrics)
    test_metrics = torch.from_numpy(test_metrics)
    '''
    train_toPredict = torch.from_numpy(train_toPredict)
    test_toPredict = torch.from_numpy(test_toPredict)
    '''
    '''
    data = MyCoolDataset(train_metrics, test_metrics, train_toPredict,
                         test_toPredict)
    dl = DataLoader(data, batch_size=200)

    train(dl, 5)
    '''
    train(5, train_metrics, test_metrics, train_toPredict, test_toPredict)


'''
def forwardPropagate():
    return None


def updateExpectedValues():
    return None


def backwardPropagate():
    return None


def updateWeights():
    pass

'''
'''
def train(trainloader, n_epochs):
'''


def train(n_epochs, x_train, x_test, y_train, y_test):
    mlp = MLPModel(2, 10)
    loss_function = CrossEntropyLoss()
    optimizer = SGD(mlp.parameters(), lr=1e-4)

    mlp.eval()
    y_pred = mlp(x_test)
    before_train = loss_function(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())
    '''
    for epoch in range(n_epochs):
        current_loss = 0

        #iterate over DataLoader (??)
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data

            optimizer.zero_grad()
            outputs = mlp(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()

            optimizer.step()
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

        # Process is complete.

    print('Training process has finished.')
    '''
    '''
    outputs = forwardPropagate()
    expected_values = updateExpectedValues()
    g = backwardPropagate()
    updateWeights(network, g)
    '''


main()
