import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAtt_RNN(nn.Module):
    def __init__(self, num_classes, in_channel=1, hidden_dim=128, n_head=4):
        super(MHAtt_RNN, self).__init__()
        self.num_classes = num_classes
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channel, 10, (5,1), stride=(1,1), dilation=(1,1), padding='same'),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 1, (5,1), stride=(1,1), dilation=(1,1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(2),
        )
        
        # self.rnn = nn.LSTM(40,self.hidden_dim, num_layers=2, bidirectional=True, 
        #                      batch_first=True)
        self.rnn = nn.GRU(40, self.hidden_dim, num_layers=2, bidirectional=True, 
                  batch_first=True)
        self.q_emb = nn.Linear(self.hidden_dim<<1, (self.hidden_dim<<1)*self.n_head)
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(256*n_head,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.Linear(32,self.num_classes)
        )
        #self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        # [B, 1, F, T]
        x = x.transpose(2, 3)
        batch_size = x.size(0)
        # [B, 1, T, F]
        x = self.cnn_extractor(x)

        # x = x.reshape(x.size(0),-1,x.size(1))
        # [B, 1, T, F]
        batch, channels, time, features = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch, time, -1)

        # [B, T, F]
        x,_ = self.rnn(x)

        middle = x.size(1)//2
        mid_feature = x[:,middle,:]

        multiheads = []
        queries = self.q_emb(mid_feature).view(self.n_head, batch_size, -1, self.hidden_dim<<1)
        for query in queries:
            att_weights = torch.bmm(query,x.transpose(1, 2))
            att_weights = F.softmax(att_weights, dim=-1)
            multiheads.append(torch.bmm(att_weights, x).view(batch_size,-1))
        x = torch.cat(multiheads, dim=-1)
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x
