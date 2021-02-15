# ----------------------------------------------------------------Importing the Libraries----------------------------------------------------------------
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torchvision import transforms
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.functional as F
import os
from tqdm import tqdm
from time import sleep

# ----------------------------------------------------------------Hyper Parameters ----------------------------------------------------------------------

batch_size = 32
num_epochs = 100
learning_rate = 3e-4
num_classes = 10
checkpoint_file = 'checkpoints\\checkpoint{}.pth.tar'
load_model    = True if os.path.isfile("my_checkpoint.pth.tar") else False
load_model = False
device_ = 'cuda'if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------Data Transformations-------------------------------------------------------------------
data_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
        std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    )
])
# ----------------------------------------------------------------Download the dataset----------------------------------------------------------------

# Download the files mannually because it was taking time 
cifar10_train = datasets.CIFAR10(root = 'data', transform = data_transforms, train =True, download = True)
cifar10_val = datasets.CIFAR10(root = 'data', transform = data_transforms, train = False, download = True)

# ----------------------------------------------------------------Info about the dataset----------------------------------------------------------------

print(type(cifar10_train).__mro__)
print(cifar10_val._check_integrity)
print(cifar10_val.classes)

# ----------------------------------------------------------------Creating DataLoaders-----------------------------------------------------------------

train_loader = DataLoader(dataset = cifar10_train, batch_size = batch_size, shuffle = True)
val_loader  = DataLoader(dataset = cifar10_val, batch_size = batch_size, shuffle = True)

# ----------------------------------------------------------------Create the model---------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=num_classes):
        super(Net, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=in_channels, out_channels = 512, kernel_size =3, padding = (1,1))
        self.conv2  = nn.Conv2d(in_channels=512, out_channels = 256, kernel_size =3, padding = (1,1))  
        self.conv3  = nn.Conv2d(in_channels=256, out_channels = 128, kernel_size =3, padding = (1,1))     
        self.relu   = nn.ReLU()
        self.pool1  = nn.MaxPool2d(kernel_size=2)
        self.pool2  = nn.MaxPool2d(kernel_size=2)
        self.fc1    = nn.Linear(in_features= 128 * 8 * 8, out_features = 64)
        self.fc2    = nn.Linear(in_features = 64, out_features = num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out.view(-1, 128 * 8 * 8)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ----------------------------------------------------------------Test Run----------------------------------------------------------------------------

def test_model():
    x = torch.rand((1,3,32,32))
    model = Net().to(device_)
    out = model(x)
    print(out.shape)
    del model
    print("Test Succesfull")


# ----------------------------------------------------------------Initialize Model----------------------------------------------------------------

model =  Net().to(device_)

# ----------------------------------------------------------------Declare Optimizer and Loss Function--------------------------------------------------

loss_function = nn.CrossEntropyLoss()
optimizer     = torch.optim.SGD(params = model.parameters(), lr = learning_rate)

# pro tip : to check argument of any functions
  
"""import inspect
inspect.signature(functionname)
"""

# ----------------------------------------------------------------Define Train Function----------------------------------------------------------------

def train():
    total_inputs = 0
    total_loss = 0
    correct = 0
    for epoch in range(num_epochs):
        model.train() #Declaring the training mode
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
        for batch_idx, (data, target) in loop:
            data, target = data.to(device_), target.to(device_)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total_loss +=loss.item()
            total_inputs += len(target)
            _, predicted = torch.max(output, dim=1)
            correct += int((predicted == target).sum())
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = total_loss/total_inputs, accu = "{0:.0%}".format(correct/total_inputs))            

    print("training Completed !!")


# ----------------------------------------------------------------Define Validation Function------------------------------------------------------------

def val():
    pass

# ---------------------------------------------------------------- Run IF Main --------------------------------------------------------------------------

if __name__ =='__main__':
    train()
    val()