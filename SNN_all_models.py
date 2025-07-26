import torch
import torch.nn as nn
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import utils, spikegen, surrogate
from snntorch import functional as SF
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform, this grayscales the MNIST dataset & resizes it as well as normalizes it so the colours are between 0 & 1
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST("./data", train=True, download=False, transform=transform)
mnist_test = datasets.MNIST("./data", train=False, download=False, transform=transform)


print(device)

# Network Architecture
num_inputs = 28*28
num_outputs = 10

spike_grad = surrogate.fast_sigmoid(slope=25)

# Define Network
class Net(nn.Module):
    def __init__(self, num_steps, beta, h_layers):
        super().__init__()

        self.num_steps = num_steps
        
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, h_layers)
        self.lif1 = snn.Leaky(beta=beta) #threshold is automatically set to 1
        self.fc2 = nn.Linear(h_layers, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    
class PopNet(nn.Module):
    def __init__(self, num_steps, beta, h_layers):
        super().__init__()

        self.num_steps = num_steps
        population = 80
        
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, h_layers)
        self.lif1 = snn.Leaky(beta=beta) #threshold is automatically set to 1
        self.fc2 = nn.Linear(h_layers, num_outputs*population)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Define Network
class ConvNet(nn.Module):
    def __init__(self, num_steps, beta):
        super().__init__()

        self.num_steps = num_steps

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 8, 5, padding="same")
        self.lif1 = snn.Leaky(beta=beta)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 24, 5, padding="same")
        self.lif2 = snn.Leaky(beta=beta)
        self.mp2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(7*7*24, 10)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(cur1), mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(cur2), mem2)
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

# Define Network
class SurConvNet(nn.Module):
    def __init__(self, num_steps, beta):
        super().__init__()

        self.num_steps = num_steps
        

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 8, 5, padding="same")
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 24, 5, padding="same")
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.mp2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(7*7*24, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(cur1), mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(cur2), mem2)
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

# Load the network onto CUDA if available



def data_output(net, data, targets, counter, iter_counter, epoch, type):
    if type == "Conv" or type == "S-Conv":
        output, _ = net(data)
    else:
        output, _ = net(data.flatten(1))
    
    if type == "Pop":
        acc = SF.accuracy_rate(output, targets, population_code=True, num_classes=10)
    #elif type == "S-Conv":
    #    acc = SF.accuracy_rate(output, targets)
    else:
        _, max_spike = output.sum(dim=0).max(1)
        acc = np.mean((targets == max_spike).detach().cpu().numpy())
    acc = acc*100

    acc_list.append(acc)
    if counter % 50 == 0:
        print(f"Epoch {epoch}, Iteration {iter_counter}")
        print(f"Train Set Loss: {loss_hist[counter]:.2f}")
        print(f"Train set accuracy for a single minibatch: {acc:.2f}%")
        print()



# These hold the accuracy & loss of the model
loss_hist = []
acc_list = []

#Holds spike data
spk_records = []

def train_network(num_epoch, batch_size, hidden, lr, time_steps, beta, type):

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

    loss = nn.CrossEntropyLoss()

    if type == "FC":
        net = Net(time_steps, beta, hidden).to(device)
    elif type == "Pop":
        net = PopNet(time_steps, beta, hidden).to(device)
        loss = SF.ce_count_loss(population_code=True, num_classes=10)
    elif type == "S-Conv":
        net = SurConvNet(time_steps, beta).to(device)
        #loss = SF.ce_rate_loss()
    else:
        net = ConvNet(time_steps, beta).to(device)

    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    counter = 0

    # Outer training loop
    for epoch in tqdm(range(num_epoch)):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()

            if type == "Conv" or type == "S-Conv":
                info = data
            else:
                info = data.flatten(1) #or data.view(batch_size, -1)

            #print(info.size())
            spike_rec , mem_rec = net(info)

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            if type == "Pop":# or type == "S-Conv":
                loss_val = loss(spike_rec, targets)
            else:
                for step in range(time_steps):
                    #print(step)
                    loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Print train/test loss/accuracy
            
            data_output(net, data, targets, counter, iter_counter, epoch, type)

            counter += 1
            iter_counter +=1

    return net


def model_accuracy(model, b_size, type):
    total = 0
    correct = 0
    running_accuracy = 0

    test_loader = DataLoader(mnist_test, batch_size=b_size, shuffle=True)

    with torch.no_grad():
        model.eval()

        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            if type == "Conv" or type == "S-Conv":
                info = data
            else:
                info = data.flatten(1) #or data.view(batch_size, -1)

            # forward pass
            test_spk, _ = model(info)

            if type == "Pop":
                running_accuracy += SF.accuracy_rate(test_spk, targets, population_code=True, num_classes=10)
                
            #if type == "S-Conv":
            #    running_accuracy += SF.accuracy_rate(test_spk, targets)
                   
            else:
                # calculate total accuracy
                _, predicted = test_spk.sum(dim=0).max(1)
                correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        if type == "Pop":# or type == "S-Conv":            
            accuracy = running_accuracy*100 / total
            print(f"Test Set Accuracy: {accuracy*100:.2f}%")
        else:
            print(f"Total correctly classified test set images: {correct}/{total}")
            print(f"Test Set Accuracy: {100 * correct / total:.2f}%")


def forward_pass(net, num_steps, batch_size, type):

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    if type == "Conv" or type == "S-Conv":
        info = data
    else:
        info = data.flatten(1)

    for step in range(num_steps):
        spk_out, mem_out = net(info)

    return spk_out, mem_out, targets

print("---------------------")
print("Fully Connected SNN")
print("---------------------")

labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']

#HYPERPARAMETERS SET 1
batch_size = 128
num_hidden = 1000
num_steps = 15
beta = 0.90 #this is the decay so at every time step there will be a 10% decrease of the membrane potential
lr = 0.0001
num_epochs = 1


model = train_network(num_epochs, batch_size, num_hidden, lr, num_steps, beta, "FC")
model_accuracy(model, batch_size, "FC")
spk_rec, mem_rec, actual = forward_pass(model, num_steps, batch_size, "FC")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

# Plot Loss & Accuracy 
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

print()

loss_hist.clear()
acc_list.clear()

print("-----------------")

#HYPERPARAMETERS SET 2
batch_size_2 = 256
num_hidden_2 = 500
num_steps_2 = 30
beta_2 = 0.96 
lr_2 = 0.00005
num_epochs_2 = 3

model_2 = train_network(num_epochs_2, batch_size_2, num_hidden_2, lr, num_steps_2, beta_2, "FC")
model_accuracy(model_2, batch_size_2, "FC")
spk_rec, mem_rec, actual = forward_pass(model_2, num_steps_2, batch_size_2, "FC")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

print()

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()

print("-----------------")

#HYPERPARAMETERS SET 3
batch_size_3 = 64
num_hidden_3 = 750
num_steps_3 = 65
beta_3 = 0.82 
lr_3 = 0.00005
num_epochs_3 = 3

model_3 = train_network(num_epochs_3, batch_size_3, num_hidden_3, lr, num_steps_3, beta_3, "FC")
model_accuracy(model_3, batch_size_3, "FC")
spk_rec, mem_rec, actual = forward_pass(model_3, num_steps_3, batch_size_3, "FC")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()

print("---------------------")
print("Fully Connected SNN w/ Population Coding")
print("---------------------")

p_model = train_network(num_epochs, batch_size, num_hidden, lr, num_steps, beta, "Pop")
model_accuracy(p_model, batch_size, "Pop")
spk_rec, mem_rec, actual = forward_pass(p_model, num_steps, batch_size, "Pop")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

#fig = plt.figure(facecolor="w", figsize=(10, 5))
#ax = fig.add_subplot(111)
#anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, interpolate=4)
print(f"The target label is: {actual[0]}")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()

print("-----------------")

p_model_2 = train_network(num_epochs_2, batch_size_2, num_hidden_2, lr, num_steps_2, beta_2, "Pop")
model_accuracy(p_model_2, batch_size_2, "Pop")
spk_rec, mem_rec, actual = forward_pass(p_model_2, num_steps_2, batch_size_2, "Pop")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

#fig = plt.figure(facecolor="w", figsize=(10, 5))
#ax = fig.add_subplot(111)
#anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, interpolate=4)
print(f"The target label is: {actual[0]}")
print()

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()

print("---------------------")
print("Convoluted SNN")
print("---------------------")

c_model = train_network(num_epochs, batch_size, num_hidden, lr, num_steps, beta, "Conv")
model_accuracy(c_model, batch_size, "Conv")
spk_rec, mem_rec, actual = forward_pass(c_model, num_steps, batch_size, "Conv")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

# Plot Loss & Accuracy 
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

print()

loss_hist.clear()
acc_list.clear()

print("-----------------")

c_model_2 = train_network(num_epochs_2, batch_size_2, num_hidden_2, lr, num_steps_2, beta_2, "Conv")
model_accuracy(c_model_2, batch_size_2, "Conv")
spk_rec, mem_rec, actual = forward_pass(c_model_2, num_steps_2, batch_size_2, "Conv")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")


fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()



print("---------------------")
print("Convoluted SNN w/ Surrogate Gradient Descent")
print("---------------------")

s_c_model = train_network(num_epochs, batch_size, num_hidden, lr, num_steps, beta, "S-Conv")
model_accuracy(s_c_model, batch_size, "S-Conv")
spk_rec, mem_rec, actual = forward_pass(s_c_model, num_steps, batch_size, "S-Conv")
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)

anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

loss_hist.clear()
acc_list.clear()

print("---------------------")

s_c_model_2 = train_network(num_epochs_2, batch_size_2, num_hidden_2, lr_2, num_steps_2, beta_2, "S-Conv")
model_accuracy(s_c_model_2, batch_size_2, "S-Conv")
spk_rec, mem_rec, actual = forward_pass(s_c_model_2, num_steps_2, batch_size_2, "S-Conv")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spk_rec[:, 0].detach().cpu(), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
anim = splt.spike_count(spk_rec[:, 0].detach().cpu(), fig, ax, labels=labels, interpolate=4)
print(f"The target label is: {actual[0]}")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(acc_list)
plt.title("Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")

#loss_hist.clear()
#acc_list.clear()

plt.show()
