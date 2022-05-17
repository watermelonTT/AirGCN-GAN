# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scaled_Laplacian, cheb_polynomial

class AuxiliaryNet_Spatial(nn.Module):


    def __init__(self, num_of_vertices, num_of_timesteps, in_channels, biDirectional = False, num_layers = 1, tau=1):
        super(AuxiliaryNet_Spatial, self).__init__()
        self.hidden_size = num_of_vertices
        self.embedding_length = num_of_timesteps * in_channels	
        self.biDirectional	= biDirectional
        self.num_layers = num_layers

        self.aux_lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional = self.biDirectional, num_layers = self.num_layers, batch_first = True)   # Dropout  
        if(self.biDirectional):
            self.aux_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.aux_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tau = tau
    

    def forward(self, input_sequence, is_train = True, batch_size=None):

        input_sequence_1 = input_sequence.reshape([input_sequence.shape[0], input_sequence.shape[1], -1]).transpose(2,1)
        out_lstm, (final_hidden_state, final_cell_state) = self.aux_lstm(input_sequence_1)
        out_linear = self.aux_linear(out_lstm)
        p_t = self.sigmoid(out_linear)

        if is_train:
            p_t = p_t.unsqueeze(-1)
            p_t = p_t.repeat(1,1,1,2)
            p_t[:,:,:,0] = 1 - p_t[:,:,:,0] 
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)   
            g_t = g_hat[:,:,:,1]
        else:
            # size : same as p_t [ batch_size x seq_len x 1]
            m = torch.distributions.bernoulli.Bernoulli(p_t)   
            g_t = m.sample()
            gt_sum = g_t.sum(1)
            x = (gt_sum == 0).nonzero(as_tuple=False)
            x = x[:,0]
            for i in x:
                g_t[i,:,:] = torch.ones(g_t[i,:,:].shape)
        
        g_t = g_t.transpose(2,1).reshape([input_sequence.shape[0], input_sequence.shape[1], input_sequence.shape[2], input_sequence.shape[3]])

        return g_t


class AuxiliaryNet_Temporal(nn.Module):
    """
    Arguments
    ---------
    batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
    aux_hidden_size : Size of the hidden_state of the LSTM   (* Later BiLSTM, check dims for BiLSTM *)
    embedding_length : Embeddding dimension of GloVe word embeddings
    --------
    """
    def __init__(self, num_of_vertices, num_of_timesteps, in_channels, biDirectional = False, num_layers = 1, tau=1):
        super(AuxiliaryNet_Temporal, self).__init__()
        self.hidden_size = num_of_timesteps
        self.embedding_length = num_of_vertices * in_channels	
        self.biDirectional	= biDirectional
        self.num_layers = num_layers

        self.aux_lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional = self.biDirectional, num_layers = self.num_layers, batch_first = True)   # Dropout  
        if(self.biDirectional):
            self.aux_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.aux_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tau = tau
    

    def forward(self, input_sequence, is_train = True, batch_size=None):
      
        # input : Dimensions (batch_size x seq_len x embedding_length)
        input_sequence_1 = input_sequence.reshape([input_sequence.shape[0], -1, input_sequence.shape[-1]])
        out_lstm, (final_hidden_state, final_cell_state)  = self.aux_lstm(input_sequence_1)    # ouput dim: ( batch_size x seq_len x hidden_size ) 
        out_linear = self.aux_linear(out_lstm)                                               # p_t dim: ( batch_size x seq_len x 1)
        p_t = self.sigmoid(out_linear)

        if is_train:
            p_t = p_t.unsqueeze(-1)
            p_t = p_t.repeat(1,1,1,2)
            p_t[:,:,:,0] = 1 - p_t[:,:,:,0]
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)   
            g_t = g_hat[:,:,:,1]
      
        else:
            # size : same as p_t [ batch_size x seq_len x 1]
            m = torch.distributions.bernoulli.Bernoulli(p_t)   
            g_t = m.sample()
            gt_sum = g_t.sum(1)
            x = (gt_sum == 0).nonzero(as_tuple=False)
            x = x[:,0]
            for i in x:
                g_t[i,:,:] = torch.ones(g_t[i,:,:].shape)
        g_t = g_t.reshape([input_sequence.shape[0], input_sequence.shape[1], input_sequence.shape[2], input_sequence.shape[3]])
        return g_t


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

        self.auxiliary = AuxiliaryNet_Spatial(num_of_vertices, num_of_timesteps, in_channels).to(DEVICE)

    def masked_Softmax(self, logits, mask):
            mask_bool = mask >0
            logits[~mask_bool] = float('-inf')
            return F.softmax(logits, dim=1)	
      
    def forward(self, x, is_train = True):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        g_t = self.auxiliary(x, is_train)

        x = x * g_t

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  

        product = torch.matmul(lhs, rhs) 

        S = torch.matmul(self.Vs, torch.tanh(product + self.bs))  

        S_normalized = F.softmax(S, dim=1)    

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int make_model(DEVI
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

        self.auxiliary = AuxiliaryNet_Temporal(num_of_vertices, num_of_timesteps, in_channels).to(DEVICE)

    def masked_Softmax(self, logits, mask):
        mask_bool = mask >0
        logits[~mask_bool] = float('-inf')
        return F.softmax(logits, dim=1)	   
    
    def forward(self, x, is_train = True):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        g_t = self.auxiliary(x, is_train)

        x = x * g_t

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.tanh(product + self.be))  # (B, T, T) 

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class Generator_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(Generator_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter) 

    def forward(self, x, is_train = True):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x, is_train)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x, is_train)  

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x_TAt, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual

class Generator_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(Generator_submodule, self).__init__()

        self.BlockList = nn.ModuleList([Generator_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)])

        self.BlockList.extend([Generator_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, is_train = True):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x, is_train)


        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


class Generator(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, cheb_polynomials, num_for_predict, num_of_vertices):
              

        super(Generator, self).__init__()

        self.SubmoduleList = nn.ModuleList([Generator_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_for_predict, 12, num_of_vertices)]) # week(time_strides = 1, len_input = 12)

        self.SubmoduleList.append(Generator_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_for_predict, 12, num_of_vertices)) # day(time_strides = 1, len_input = 12)

        self.SubmoduleList.append(Generator_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 3, cheb_polynomials, num_for_predict, 36, num_of_vertices)) # hour(time_strides = 3, len_input = 36)
        
        self.fusion_week = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_for_predict).to(DEVICE)) #

        self.fusion_day = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_for_predict).to(DEVICE)) #

        self.fusion_hour = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_for_predict).to(DEVICE)) #
       

        self.combine = nn.Sequential(
            nn.Linear(in_features= 24,out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=12)
        )

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x, noise_batch, is_train = True):
       
        submodule_output_week = self.SubmoduleList[0](x[:,:,:,0:12], is_train)    # week

        submodule_output_day = self.SubmoduleList[1](x[:,:,:,12:24], is_train)   # day
        
        submodule_output_hour = self.SubmoduleList[2](x[:,:,:,24:60], is_train)   # hour

        submodule_output = submodule_output_week * self.fusion_week + submodule_output_day * self.fusion_day + submodule_output_hour * self.fusion_hour

        combine_input = torch.cat((submodule_output, noise_batch), dim=2)

        output = self.combine(combine_input)
       
        return output

class Discriminator(nn.Module):
    def __init__(self, DEVICE, num_of_vertices, num_of_weeks, num_of_days, mean, std): 
        super(Discriminator, self).__init__()
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.mean = mean
        self.std = std
        self.DEVICE = DEVICE
        self.to(DEVICE)

        self.model_extract = nn.Sequential(
                        nn.Conv1d(in_channels=num_of_vertices, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1),
                        nn.LeakyReLU(),
                        nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                        nn.LeakyReLU(),
                        nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                        nn.LeakyReLU(),
                        nn.LSTM(input_size=24, hidden_size=16, num_layers=3, batch_first = True)
        )

        self.model_flatten = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=512, out_features=256),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=256, out_features=256),
                    nn.ReLU(),
                    nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, prediction, condition):
        condition = condition[:, :, 0, ((self.num_of_weeks + self.num_of_days) * 12):60] 
        d_input = torch.cat((condition, prediction), dim = 2)
        d_input = (d_input - self.mean) / self.std
        d_latent, _ = self.model_extract(d_input) 
        output = self.model_flatten(d_latent) 
        return output

def make_Generator(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, adj_mx, num_for_predict, num_of_vertices):
    '''

    :param DEVICE:
    :param nb_block:3
    :param in_channels:1
    :param K:3
    :param nb_chev_filter:64
    :param nb_time_filter:64
    :param time_strides:3
    :param cheb_polynomials:
    :param nb_predict_step:12
    :param len_input：12
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = Generator(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, cheb_polynomials, num_for_predict, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model

def make_Discriminator(DEVICE, num_of_vertices, num_of_weeks, num_of_days, mean=0, std=1):
    '''
    :param DEVICE:
    :param num_of_vertices:933
    :param num_of_weeks:1
    :param num_of_days:1
    :param mean:
    :param std:
    '''
    model = Discriminator(DEVICE, num_of_vertices, num_of_weeks, num_of_days, mean, std)

    return model