import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(in_features=50176, out_features=4096)
        self.fc2   = nn.Linear(in_features=4096, out_features=4096)
        # MODIF : J'ai modifié la sortie de la couche ci-dessous de 10 vers 4 (pour 4 classes)
        self.fc3   = nn.Linear(in_features=4096, out_features=4)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(3, 3), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(3, 3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # toutes les dimensions sauf celle du batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features