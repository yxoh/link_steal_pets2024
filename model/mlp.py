import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP_ATTACK(nn.Module):
    def __init__(self, dim_in):
        super(MLP_ATTACK, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc3 = nn.Linear(32, 2) 

    def forward(self, x):
        x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.relu(self.fc3(x))
        return out


class MLP_ATTACK_PLUS(nn.Module):
    def __init__(self, dim_in_1, dim_in_2):
        super(MLP_ATTACK_PLUS, self).__init__()
        self.dim_in_1 = dim_in_1
        self.dim_in_2 = dim_in_2

        self.fc1 = nn.Linear(self.dim_in_1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        

        self.fc4 = nn.Linear(self.dim_in_2, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(32, 2)
        
    
    def forward(self, x1, x2):
        x1 = x1.view(-1, self.dim_in_1)
        x2 = x2.view(-1, self.dim_in_2)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x2 = F.relu(self.fc4(x2))
        x2 = F.relu(self.fc5(x2))
        
        combine = torch.cat([x1, x2], dim=1)
        out = F.relu(self.fc6(combine))

        return out
    

class MLP_ATTACK_PLUS2(nn.Module):
    def __init__(self, dim_in_1, dim_in_2):
        super(MLP_ATTACK_PLUS2, self).__init__()
        self.dim_in_1 = dim_in_1
        self.dim_in_2 = dim_in_2

        self.fc1 = nn.Linear(self.dim_in_1, 16)
        self.fc2 = nn.Linear(16, 4)

        self.fc3 = nn.Linear(self.dim_in_2, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)

        self.fc6 = nn.Linear(20, 2)
        
    
    def forward(self, x1, x2):
        x1 = x1.view(-1, self.dim_in_1)
        x2 = x2.view(-1, self.dim_in_2)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))

        x2 = F.relu(self.fc3(x2))
        x2 = F.relu(self.fc4(x2))
        x2 = F.relu(self.fc5(x2))
        
        combine = torch.cat([x1, x2], dim=1)
        out = F.relu(self.fc6(combine))

        return out
    

class MLP_ATTACK_ALL(nn.Module):
    def __init__(self, dim_in_1, dim_in_2, dim_in_3):
        super(MLP_ATTACK_ALL, self).__init__()
        self.dim_in_1 = dim_in_1
        self.dim_in_2 = dim_in_2
        self.dim_in_3 = dim_in_3

        self.fc1 = nn.Linear(self.dim_in_1, 128) 
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 16)

        self.fc4 = nn.Linear(self.dim_in_2, 128) 
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 16)

        self.fc7 = nn.Linear(self.dim_in_3, 4)

        self.fc8 = nn.Linear(36, 2)
        
    
    def forward(self, x1, x2, x3):
        x1 = x1.view(-1, self.dim_in_1)
        x2 = x2.view(-1, self.dim_in_2)
        x3 = x3.view(-1, self.dim_in_3)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))

        x2 = F.relu(self.fc4(x2))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))


        x3 = F.relu(self.fc7(x3))

        combine = torch.cat([x1, x2, x3], dim=1)
        out = F.relu(self.fc8(combine))

        return out


class MLP_Target(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_Target, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.fc1 = nn.Linear(dim_in, 32)
        self.fc2 = nn.Linear(32, dim_out)

    def forward(self, x):
        x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
