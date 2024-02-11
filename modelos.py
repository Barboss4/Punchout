import torch
import torch.nn as nn
import torch.nn.functional as F

# Defina o modelo DQN
class DQN(torch.nn.Module):
    def __init__(self,entrada_dim, output_dim,kernel):
        saida_dim = output_dim 
        super().__init__()
        self.camada_entrada = nn.Conv2d(entrada_dim, 100, kernel_size=kernel)
        self.camada_oculta = nn.Conv2d(100, 25,kernel_size=kernel)
        self.camada_saida = nn.Conv2d(25, saida_dim,kernel_size=kernel)

        linear_input_size = 2156
        
        self.fc1 = nn.Linear(linear_input_size,500)
        self.fc2 = nn.Linear(500, 125)
        self.fc3 = nn.Linear(125, 40)
        self.fc4 = nn.Linear(40, output_dim)
    
    def forward(self, x):
        x = F.tanh(self.camada_entrada(x))
        x = F.tanh(self.camada_oculta(x))
        x = F.tanh(self.camada_saida(x))      
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        outputmodelo = self.fc4(x)
        return outputmodelo

