import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
import math

# invert + intra

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.projector = nn.Linear(configs.d_model ** 2, configs.pred_len, bias=True)

        # patch paraments
        self.num_nodes = configs.num_nodes
        
        # PiFormer2
        self.width = configs.width
        self.length = configs.length
        self.patch_width = configs.patch_width
        self.patch_length = configs.patch_length
        self.patch_size = self.patch_width * self.patch_length
        self.patch_nums = int(self.num_nodes / self.patch_size)
        self.patch_width_nums = int(self.width / self.patch_width)
        self.patch_length_nums = int(self.length / self.patch_length)

        # intra_patch_attention
        self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, configs.d_model, 16), requires_grad=True)
        self.embedding_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(16, configs.d_model)]) for _ in range(self.patch_nums)
        ])
        self.intra_d_model = configs.d_model
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=True)
        self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16, num_nodes=configs.d_model,
                                                          factorized=True, number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None, num_nodes=configs.d_model,
                                                        factorized=False, number_of_weights=2)
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)

        # FFN
        self.d_ff = configs.d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(configs.d_model, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_ff, configs.d_model, bias=True))
        


    def forecast(self, x_enc):

        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        _, _, N = x_enc.shape # B L N

        # Embedding
        # B L N -> B N E E
        enc_out = self.enc_embedding(x_enc, None)

        new_enc_out = enc_out
        batch_size = enc_out.size(0)
        intra_out_concat = None

        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        for i in range(self.patch_length_nums):
            p_length_start = i * self.width * self.patch_length
            for j in range(self.patch_width_nums):
                p_width_start = j * self.patch_width
                t = []
                for k in range(self.patch_length):
                    p_start = p_length_start + k * self.width + p_width_start
                    p_end = p_start + self.patch_width
                    temp = enc_out[:, p_start : p_end, :, :]
                    t.append(temp)
                t = torch.cat(t, dim=1)


                intra_emb = self.embedding_generator[i](self.intra_embeddings[i]).expand(batch_size, -1, -1, -1)
                t = torch.cat([intra_emb, t], dim=1)
                out, attention = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct, weights_shared,
                                                            biases_shared)

                if intra_out_concat == None:
                    intra_out_concat = out
                else:
                    intra_out_concat = torch.cat([intra_out_concat, out], dim=1)

        intra_out_concat = intra_out_concat.permute(0,3,2,1)
        intra_out_concat = self.intra_Linear(intra_out_concat)
        intra_out_concat = intra_out_concat.permute(0,3,2,1)

        intra_out_concat = F.pad(intra_out_concat, (0, 0, 0, 0, 0, 1), mode='constant', value=0)

        out = new_enc_out + intra_out_concat

        ##FFN
        out = self.dropout(out)
        out = self.ff(out) + out

        # B N E E -> B N E^2
        out = out.reshape(batch_size, self.num_nodes, -1)
        # B N E^2 -> B N S -> B S N 
        dec_out = self.projector(out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    

class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x, attention


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B
        