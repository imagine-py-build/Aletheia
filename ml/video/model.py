import torch
from torch import nn
class CNNGRU(nn.Module):
    def __init__(self,feature_dim=128):
        super().__init__(); self.gru=nn.GRU(feature_dim,128,batch_first=True); self.fc=nn.Linear(128,2)
    def forward(self,x): out,_=self.gru(x); return self.fc(out[:,-1])
