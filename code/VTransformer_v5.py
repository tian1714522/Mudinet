from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.codegen.fnodes import reshape
from torch.nn.functional import dropout
from torch.utils.hooks import RemovableHandle

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        # 定义多头的线性变换
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, x):
        batch_size, T, _ = x.size()

        Q = self.query(x)  # (batch_size, T, d_model)
        K = self.key(x)  # (batch_size, T, d_model)
        V = self.value(x)  # (batch_size, T, d_model)

        # 多头拆分: (batch_size, num_heads, T, head_dim)
        Q = Q.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, T, T)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, T, T)

        # 使用注意力权重加权值
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, T, head_dim)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, T, self.d_model)

        # 输出层
        out = self.out(attention_output)

        return out

# 自注意力子块
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, nhead,dropout_rate=0.1,attention_dropout_rate=0.1):
        super(AttentionBlock,self).__init__()

        self.dropout_rate = dropout_rate

        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.layer_norm_out = nn.LayerNorm(hidden_dim)
        self.drop_out_attention = nn.Dropout(attention_dropout_rate)


        self.attention = SelfAttention(hidden_dim, nhead)
        self.mlp = FeedForward(hidden_dim, hidden_dim, dropout_rate)

    def forward(self,inputs):
        inputs = self.layer_norm_input(inputs)
        x = self.attention(inputs)
        x = self.drop_out_attention(x)
        x = x + inputs
        y = self.layer_norm_out(x)
        y = self.mlp(y)

        return x + y

class MlpEncoder(nn.Module):
    def __init__(self,cir_dim,hidden_dim,t):
        super(MlpEncoder,self).__init__()
        self.fc_encoder = nn.Sequential(nn.Linear(cir_dim, 512),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(512, 1024),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(1024, 2048),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(2048, hidden_dim),
                                         )
    def forward(self,x):
        return self.fc_encoder(x)


class MlpDecoder(nn.Module):
    def __init__(self,hidden_dim,cir_dim,t):
        super(MlpDecoder,self).__init__()
        self.fc_decoder = nn.Sequential(nn.Linear(hidden_dim, 512),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(512, 512),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(512, 512),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(512, cir_dim),
                                         )
    def forward(self,x):
        return self.fc_decoder(x)

# 编码器
class Encoder(nn.Module):
    def __init__(self, cir_dim, hidden_dim, nhead, num_layers, global_latent_dim1, global_latent_dim2, local_latent_dim,t):
        super(Encoder, self).__init__()

        # 把cir投影到嵌入空间，以保证维度可控
        self.embedding = nn.Linear(cir_dim,hidden_dim)

        self.layers = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=nhead,dim_feedforward=1024,batch_first=True,dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.layers,num_layers=num_layers)
        # self.encoder = MlpEncoder(cir_dim,hidden_dim)

        # 第一个全局隐变量
        self.fc_global1_mu = nn.Linear(hidden_dim, global_latent_dim1)
        self.fc_global1_logvar = nn.Linear(hidden_dim, global_latent_dim1)

        # 第二个全局隐变量
        self.fc_global2_mu = nn.Linear(hidden_dim, global_latent_dim2)
        self.fc_global2_logvar = nn.Linear(hidden_dim, global_latent_dim2)

        # 局部隐变量
        self.fc_local_mu = MlpEncoder(cir_dim, local_latent_dim,t)
        self.fc_local_logvar = MlpEncoder(cir_dim, local_latent_dim,t)

        self.relu = nn.ReLU()


    def forward(self, input):
        x = self.embedding(input)
        x = self.encoder(x)

        h_mean = x.mean(dim=1)  # (batch_size, hidden_dim)

        global1_mu = self.fc_global1_mu(h_mean)
        global1_logvar = self.fc_global1_logvar(h_mean)

        global2_mu = self.fc_global2_mu(h_mean)
        global2_logvar = self.fc_global2_logvar(h_mean)

        local_mu = self.fc_local_mu(input)
        local_logvar = self.fc_local_logvar(input)


        return global1_mu, global1_logvar, global2_mu, global2_logvar, local_mu, local_logvar


# 解码器
class Decoder(nn.Module):
    def __init__(self, cir_dim, hidden_dim, nhead, num_layers, global_latent_dim1, global_latent_dim2, local_latent_dim,t):
        super(Decoder, self).__init__()
        self.reverse_embbeding=nn.Linear(hidden_dim,cir_dim)

        self.layers = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=nhead,dim_feedforward=1024,batch_first=True,dropout=0.1)
        self.decoder = nn.TransformerEncoder(self.layers,num_layers=num_layers)

        self.fc_input_projection = nn.Linear(global_latent_dim1 + global_latent_dim2 + local_latent_dim, hidden_dim)

        # self.mlpDecoder = MlpDecoder(global_latent_dim1 + global_latent_dim2 + local_latent_dim, cir_dim)


    def forward(self, global1_z, global2_z, local_z):
        batch_size, t, local_latent_dim = local_z.shape

        global1_z_expanded = global1_z.unsqueeze(1).expand(-1, t, -1)
        global2_z_expanded = global2_z.unsqueeze(1).expand(-1, t, -1)

        z = torch.cat([global1_z_expanded, global2_z_expanded, local_z], dim=-1)

        z = self.fc_input_projection(z)
        z = self.decoder(z)
        z = self.reverse_embbeding(z)

        # z = self.mlpDecoder(z)


        return z


class PositioningNetwork(nn.Module):
    def __init__(self, global_latent_dim, local_latent_dim, hidden_dim, nhead, num_layers, t, pos_dim=2):
        super(PositioningNetwork, self).__init__()
        self.fc_input_projection = nn.Linear(global_latent_dim + local_latent_dim, hidden_dim)

        self.layers = nn.ModuleList([SelfAttention(hidden_dim, nhead) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(t)
        self.fc_position = nn.Sequential(nn.Linear(hidden_dim, 2048),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(2048, 1024),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(1024, 512),
                                         nn.Dropout(0.1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(t),
                                         nn.Linear(512, pos_dim),
                                         )


    def forward(self, global1_z,local_z):
        batch_size, t, local_latent_dim = local_z.shape



        # 将全局变量扩展至每个时间点
        global1_z_expanded = global1_z.unsqueeze(1).expand(-1, t, -1)

        # 拼接全局和局部隐变量，计算每个时刻的位置
        z = torch.cat([global1_z_expanded, local_z], dim=-1)
        z = self.relu(z)
        z = self.bn(z)
        z = self.fc_input_projection(z)
        positions = self.fc_position(z)  # (batch_size, t, pos_dim)

        return positions


class MultiTaskVT(nn.Module):
    def __init__(self, hidden_dim, nhead, num_layers, global_latent_dim1, global_latent_dim2, local_latent_dim,
                 cir_dim,t=10, pos_dim=2):
        super(MultiTaskVT, self).__init__()

        self.encoder = Encoder(cir_dim, hidden_dim, nhead, num_layers, global_latent_dim1, global_latent_dim2,
                                        local_latent_dim,t)
        self.decoder = Decoder(cir_dim, hidden_dim, nhead, num_layers, global_latent_dim1, global_latent_dim2,
                                        local_latent_dim,t)
        self.positioning_net = PositioningNetwork(global_latent_dim1, local_latent_dim, hidden_dim, nhead, num_layers,
                                                 t,pos_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码器输出
        g1_mu, g1_logvar, g2_mu, g2_logvar, l_mu, l_logvar = self.encoder(x)

        # 重参数化采样全局和局部隐变量
        g1_z = self.reparameterize(g1_mu, g1_logvar)
        g2_z = self.reparameterize(g2_mu, g2_logvar)
        l_z = self.reparameterize(l_mu, l_logvar)

        # 重建的CIR信号
        recon_x = self.decoder(g1_z, g2_z, l_z)

        # 每个时刻的定位结果
        p = self.positioning_net(g1_z, l_z)

        return recon_x, p, g1_mu, g1_logvar, g2_mu, g2_logvar, l_mu, l_logvar