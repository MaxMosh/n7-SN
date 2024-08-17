import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # MODIF : J'ai remplacé 16*5*5 par 62 puis par 61504
        self.fc1   = nn.Linear(61504, 120)
        self.fc2   = nn.Linear(120, 84)
        # MODIF : J'ai modifié la sortie de la couche ci-dessous de 10 vers 4 (pour 4 classes)
        self.fc3   = nn.Linear(84, 4)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2), stride=2)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
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