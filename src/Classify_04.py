print("USAGE:\npython3 ...Classify_04 {learning rate} {user input = test}")
# Courtesy of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys


#This version tests for best learning rate

#Hyperparamaters
lr = 0.4
epochs = 20
user_test = False

#Loss function = Cross Entropy
#Model = Relu
#Optimizer = SGD

# Handle Command Line arguements
if(len(sys.argv)>1):
    lr = float(sys.argv[1])
    if(len(sys.argv)>2):
        if(sys.argv[2].lower()=="test"):
            user_test = True


# Start a counter for timing
time_begin = time.time()

# Crete array for graph data
accuracy = []
epoch_no = []
temp = str(lr).replace(".",",")
file = "plot_"+temp

# Access training data from data folder
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Access test data from data folder
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    accuracy.append((100*correct))
    temp = (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    print("\033[91m {}\033[00m" .format(temp))

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        input_size = 784
        hidden_sizes = [128, 64]
        output_size = 10
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)


for t in range(epochs):
    print("\033[96m {}{}{}\033[00m" .format("Epoch ",t+1,"\n-------------------------------"))
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
    epoch_no.append(t+1)
total_time = '{:.0f}'.format(time.time() - time_begin)
print("\033[92m {}{}{}{}\033[00m" .format("Done Training\n","Time Taken: ",total_time," seconds"))


plt.title('Accuracy')
plt.plot(accuracy, label='Accuracy')
plt.savefig('outputs/Classify_04/'+file)
plt.show()

#----------------------------------------------------------------
#User input section
#----------------------------------------------------------------

if(user_test):

    trans = Compose([ToTensor(),])


    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


    user_input = input("Please enter a filepath:\n")

    while((user_input.lower())!= "exit"):
        image = Image.open(user_input)
        formatted_input = trans(image)
        with torch.no_grad():
            pred = model(formatted_input)
            predicted = classes[pred[0].argmax(0)]

            print(f'Classifier: "{predicted}"')
        user_input = input("Please enter a filepath:")

        
    print("Exiting....")