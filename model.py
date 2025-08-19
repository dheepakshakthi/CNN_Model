import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

class CNN_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        
        self.fc1 = torch.nn.Linear(256 * 8 * 8, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = torch.nn.ReLU()
        
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc3(x)
        return x