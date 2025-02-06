import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_regularization1(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += (param.pow(power)).mean()

            cnt+=1
    return reg_loss/cnt

def compute_regularization4(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += ((-2*param**2+1).pow(power)).mean()

            cnt += 1
    return reg_loss / cnt
def compute_regularization2(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += (1/2 * (torch.cos(torch.pi*param)+1)).mean()

            cnt += 1
    return reg_loss / cnt
def compute_regularizationg(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += (torch.exp((-power*param)**2)).mean()

            cnt += 1
    return reg_loss / cnt
def compute_regularization6(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += (-1/4 * (torch.cos(torch.pi*(param+1))+1)**2 + 1).mean()

            cnt += 1
    return reg_loss / cnt
def compute_regularization7(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += (-param.pow(power)+1).mean()

            cnt += 1
    return reg_loss / cnt
def compute_regularization3(model):
    reg_loss = 0.0
    cnt = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += ((-2 * torch.abs(param) + 1).pow(power)).mean()

            cnt += 1
    return reg_loss / cnt
    # Define the MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc1_1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc1_1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__=="__main__":
    # Hyperparameters
    batch_size = 512
    learning_rate = 0.001
    lambda_reg = 1 # Regularization strength
    power = 2

    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("loading data")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print("loading data finished")

    print("loading model and optimizer")
    model = MLP().to("cuda")
    #model.load_state_dict(torch.load('rl_distinct_d2.pt'))


    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("loading model and optimizer finished")

    num_epochs = 20
    print("training starts")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_reg = 0.0
        total_or_loss = 0.0
        for data, target in train_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            cross_entropy_loss = criterion(output, target)
            reg_loss = compute_regularization2(model)
            #print(reg_loss)
            loss = cross_entropy_loss + (lambda_reg * reg_loss)
            or_loss = cross_entropy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_reg+=lambda_reg * reg_loss * data.size(0)
            total_or_loss += or_loss * data.size(0)
        avg_loss = total_loss / len(train_dataset)
        reg_avg_loss = total_reg/len(train_dataset)
        avg_or_loss = total_or_loss/len(train_dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Reg_Loss: {reg_avg_loss:.4f}, Original Loss: {avg_or_loss:.4f}")


    # Test accuracy
    def test(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to("cuda")
                target = target.to("cuda")
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")


    test(model, test_loader)
    torch.save(model.state_dict(),"rl_binary_b512_20ep_1lamb.pt")
