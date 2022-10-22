import torch
import torch.nn as nn
import torch.nn.functional as F

class SignDetection(nn.Module):
    def __init__(self, input_size=126, layer_nodes=[128, 64,64],layers=1,out_size=5,frames=30, batch_size=128):
        super().__init__()
        self.input_size=input_size
        self.out_size=out_size
        self.hidden_size=layer_nodes[0]
        self.linear_sizes=layer_nodes
        self.frames_n=frames
        self.layers_n=layers
        self.batch_size=batch_size
        self.lstm1=nn.LSTM(self.input_size, self.hidden_size,self.layers_n, batch_first=True)
        self.lstm2=nn.LSTM(self.hidden_size, self.linear_sizes[1],self.layers_n, batch_first=True)
        self.lstm3=nn.LSTM(self.linear_sizes[1], self.linear_sizes[2],self.layers_n, batch_first=True)
        self.linear1=nn.Linear(self.linear_sizes[1],self.linear_sizes[2])
        self.linear2=nn.Linear(self.linear_sizes[2],self.out_size)
        self.hidden_layer=(torch.zeros(self.layers_n,self.batch_size,self.out_size),torch.zeros(self.layers_n,self.batch_size,self.hidden_size))
#h_0: tensor of shape (D * \text{num\_layers}, N, H_{out})(D∗num_layers,N,Hout) containing the initial hidden state for each element in the batch. Defaults to zeros if (h_0, c_0) is not provided.
#c_0: tensor of shape (D * \text{num\_layers}, N, H_{cell})(D∗num_layers,N,Hcell) containing the initial cell state for each element in the batch. Defaults to zeros if (h_0, c_0) is not provided.   
    def forward(self,X):
        X,self.hidden_layer=self.lstm1(X)
        X,self.hidden_layer=self.lstm2(F.relu(X))
        X,self.hidden_layer=self.lstm3(F.relu(X))
        X=F.relu(self.linear1(X[:,-1,:].reshape(-1,self.linear_sizes[1])))#We want the output of only the last step
        X=self.linear2(X)
        return F.softmax(X,dim=-1)