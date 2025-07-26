import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


import sinabs.from_torch
import sinabs.layers as sl

from tqdm import tqdm
#Allows progress bars for training & testing loops


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform, this grayscales the MNIST dataset & resizes it as well as normalizes it so the colours are between 0 & 1
transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])




print(device)

#Hyperparameters - SET 1
batch_size = 128
lr = 0.0001
num_epochs = 1 

#Hyperparameters - SET 2
batch_size_2 = 256
lr_2 = 0.00005
num_epochs_2 = 3

#Hyperparameters - SET 3
batch_size_3 = 64
lr_3 = 0.00005
num_epochs_3 = 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 8, 5, padding="same", bias=False)
        self.conv2 = nn.Conv2d(8, 24, 5, padding="same", bias=False)

        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)

        self.flat = nn.Flatten()

        self.out = nn.Linear(7*7*24, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)

        x = self.flat(x)
        x = self.out(x)
        return x

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, is_spiking=False, time_window=100):
        super().__init__(
            root=root, train=train, download=True, transform=transform
        )
        self.is_spiking = is_spiking
        self.time_window = time_window

    def __getitem__(self, index):
        img, target = self.data[index].unsqueeze(0) / 255, self.targets[index]
        # img is now a tensor of 1x28x28

        if self.is_spiking:
            img = (torch.rand(self.time_window, *img.shape) < img).float()

        return img, target

mnist_train = MNIST("./data", train=True, is_spiking=False)
mnist_test = MNIST("./data", train=False, is_spiking=False)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

def multi_class_accuracy(y_predicted, y_test):
    y_pred_softmax = torch.log_softmax(y_predicted, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

ann = Net().to(device)

loss_hist = []
acc_hist = []

optim = torch.optim.Adam(ann.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()



counter = 0
for epoch in tqdm(range(num_epochs)):
    iter_counter = 0
    train_batch = iter(train_loader)

    for data, target in train_batch:
        data, target = data.to(device), target.to(device)
        output = ann(data)
        ann.train()

        optim.zero_grad()

        loss_val = loss(output, target)
        acc = multi_class_accuracy(output, target)
        loss_val.backward()
        optim.step()

        loss_hist.append(loss_val.item())
        acc_hist.append(acc.item())
        if counter % 50 == 0:
            print(f"Epoch {epoch}, Iteration {iter_counter}")
            print(f"Train Set Loss: {loss_hist[counter]:.2f}")
            print(f"Train set accuracy for a single minibatch: {acc:.2f}%")
            print()
        counter += 1
        iter_counter +=1

correct_predictions = []
test_batch = iter(test_loader)

for data, target in test_batch:
    data, target = data.to(device), target.to(device)
    output = ann(data)

    # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)

    # Compute the total correct predictions
    correct_predictions.append(pred.eq(target.view_as(pred)))

correct_predictions = torch.cat(correct_predictions)
print(
    f"Classification accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%"
)

input_shape = (1, 28, 28)

sinabs_model = sinabs.from_torch.from_model(
    ann, input_shape=input_shape, batch_size=1, add_spiking_output=True, synops=False
)

print(sinabs_model.spiking_model)

# Plot Loss & Accuracy 
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_hist)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")


# Time window per sample
time_window = 100  # time steps
test_batch_size = 10

spike_mnist_test = MNIST(
    "./data", train=False, is_spiking=True, time_window=time_window
)
spike_test_loader = DataLoader(
    spike_mnist_test, batch_size=test_batch_size, shuffle=True
)

correct_predictions = []

for data, target in tqdm(spike_test_loader):
    data, target = data.to(device), target.to(device)
    data = sl.FlattenTime()(data)
    with torch.no_grad():
        output = sinabs_model(data)
        output = output.unflatten(
            0, (test_batch_size, output.shape[0] // test_batch_size)
        )

    # get the index of the max log-probability
    pred = output.sum(1).argmax(dim=1, keepdim=True)

    # Compute the total correct predictions
    correct_predictions.append(pred.eq(target.view_as(pred)))

    #if len(correct_predictions) * test_batch_size >= 300:
    #    break

    

correct_predictions = torch.cat(correct_predictions)
print(
    f"Classification accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%"
)


# Get one sample from the dataloader
img, label = spike_mnist_test[6]
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.imshow(img.sum(0)[0])


snn_output = sinabs_model(img.to(device))

#print(snn_output)
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.pcolormesh(snn_output.T.detach().cpu())

plt.ylabel("Neuron ID")
plt.yticks(np.arange(10) + 0.5, np.arange(10))
plt.xlabel("Time") 


plt.show()