"""
Code from https://github.com/theswgong/spiralnet_plus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, inter_layer_size,
                 spiral_indices, down_transform, up_transform, decode_using_gt_age, diagonal_idx, 
                 is_vae=False, age_disentanglement=False, swap_feature=False, conditional_vae=False, inter_layer=False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.is_vae = is_vae
        self.age_disentanglement = age_disentanglement
        self.decode_using_gt_age = decode_using_gt_age
        self.swap_feature = swap_feature
        self.diagonal_idx = diagonal_idx
        self.conditional_vae = conditional_vae 
        self.inter_layer = inter_layer
        self.inter_layer_size = inter_layer_size

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
                
        ##############

        if self.inter_layer:
            # use an intermediate layer of a higher dimention to which another intermediate step (for mu and logvar) and age will come from 
            
            # output layer of the encoder (higher dimention than mu & logvar)
            self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], self.inter_layer_size))
            self.relu1 = nn.ReLU()

            #########

            # # WITH TWO INTERMEDIATE STEPS

            # # intermediate layer for mu and logvar (lower dimentional - same size and mu and logvar)
            # if self.age_disentanglement:
            #     self.intermediate = nn.Linear(self.inter_layer_size, latent_size-1)
            # else:
            #     self.intermediate = nn.Linear(self.inter_layer_size, latent_size)
            # self.relu2 = nn.ReLU()

            # # first branch to get mu and logvar
            # if self.age_disentanglement:
            #     self.mu = nn.Linear(latent_size-1, latent_size-1) 
            # else:
            #     self.mu = nn.Linear(latent_size, latent_size)
            
            # if self.is_vae:
            #     if self.age_disentanglement:
            #         self.logvar = nn.Linear(latent_size-1, latent_size-1) 
            #     else:
            #         self.logvar = nn.Linear(latent_size, latent_size)

            # WITH ONE INTERMEDIATE STEPS

            self.relu2 = nn.ReLU()

            # first branch to get mu and logvar
            if self.age_disentanglement:
                self.mu = nn.Linear(self.inter_layer_size, latent_size-1) 
            else:
                self.mu = nn.Linear(self.inter_layer_size, latent_size)
            
            if self.is_vae:
                if self.age_disentanglement:
                    self.logvar = nn.Linear(self.inter_layer_size, latent_size-1) 
                else:
                    self.logvar = nn.Linear(self.inter_layer_size, latent_size)

            ##########
            
            # second branch to get age 
            if self.age_disentanglement:
                self.age = nn.Linear(self.inter_layer_size, 1) 
        
        else:
            # orginal code where encoder layer takes straight to mu and logvar
            # mu layer (if age_disentanglement is True, age latent is removed from mu layer)
            self.en_layers.append(
                nn.Linear(self.num_vert * out_channels[-1], latent_size))

            # logvar layer
            if self.is_vae:  
                if self.age_disentanglement:
                    self.en_layers.append(
                        nn.Linear(self.num_vert * out_channels[-1], latent_size-1))
                else:
                    self.en_layers.append(
                        nn.Linear(self.num_vert * out_channels[-1], latent_size))


        ##############

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_size, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        if self.conditional_vae:
            in_channels -= 1
            
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))
        self.reset_parameters() 

        # # execution for regression model 
        # self.reg_sq = nn.Sequential(
        #     nn.Linear(1, 8),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(8, 8),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(8, 1))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    ##############

    # # ORIGINAL
    # def encode(self, x):
    #     n_linear_layers = 2 if self.is_vae else 1
    #     for i, layer in enumerate(self.en_layers):
    #         if i < len(self.en_layers) - n_linear_layers:
    #             x = layer(x, self.down_transform[i])


    #     x = x.view(-1, self.en_layers[-1].weight.size(1))
    #     mu = self.en_layers[-1](x)

    #     if self.is_vae:
    #         logvar = self.en_layers[-2](x)
    #     else:
    #         mu = torch.sigmoid(mu)
    #         logvar = None
    #     return mu, logvar

    def encode(self, x):
        age = None
        if self.inter_layer:
            # NEW
            for i, layer in enumerate(self.en_layers):
                if i < len(self.en_layers) - 1:
                    x = layer(x, self.down_transform[i])
                    x = self.relu1(x)

            # reshaping to have the correct size
            x = x.view(-1, self.en_layers[-1].weight.size(1))
            z_pre = self.en_layers[-1](x)

            #########

            # # WITH TWO INTERMEDIATE STEP 

            # x_inter = self.relu2(self.intermediate(z_pre))

            # mu = self.mu(x_inter)
            # if self.is_vae:
            #     logvar = self.logvar(x_inter)
            # else:
            #     mu = torch.sigmoid(mu)
            #     logvar = None
        
            # WITH ONE INTERMEDIATE STEP

            z_pre = self.relu2(z_pre)

            mu = self.mu(z_pre)
            if self.is_vae:
                logvar = self.logvar(z_pre)
            else:
                mu = torch.sigmoid(mu)
                logvar = None

            #########


            if self.age_disentanglement:
                age = self.age(z_pre)
                mu = torch.cat((mu, age), dim=1)
        
        else:
            # ORIGINAL
            n_linear_layers = 2 if self.is_vae else 1
            for i, layer in enumerate(self.en_layers):
                if i < len(self.en_layers) - n_linear_layers:
                    x = layer(x, self.down_transform[i])

            x = x.view(-1, self.en_layers[-1].weight.size(1))
            mu = self.en_layers[-1](x)

            if self.is_vae:
                logvar = self.en_layers[-2](x)
            else:
                mu = torch.sigmoid(mu)
                logvar = None

        return mu, logvar
    
    ##############

    def decode(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x
    
    def change_age_latent(self, z, age):
        z_copy = z.clone()
        if self.swap_feature:
            z_copy = z_copy[self.diagonal_idx, ::]
            z_copy[:,-1] = age.view(-1)
            z[self.diagonal_idx, ::] = z_copy
        else:
            z[:,-1] = age.view(-1)
            
        return z
    
    # def reg_2(self, z):
    #     z = z[:, -1:]
    #     return self.reg_sq(z)   

    def forward(self, data):

        x = data.x

        # if self.conditional_vae is true, then concatinate the age onto the x data before passing it into the model
        if self.conditional_vae:
            assert self.swap_feature == False
            assert self.age_disentanglement == True
            
            age = data.norm_age

            # Expand and repeat ages to match the face_meshes shape
            age_expanded = age.unsqueeze(-1).expand(-1, x.shape[1], 1)

            # Concatenate along the last dimension
            mesh_with_age = torch.cat((x, age_expanded), dim=2)
            x = mesh_with_age.float()
        
        mu, logvar = self.encode(x)
        if self.is_vae and self.training:
            z = self._reparameterize(mu, logvar, self.age_disentanglement)
            if self.decode_using_gt_age or self.conditional_vae:
                z = self.change_age_latent(z, age)
        else:
            z = mu
    
        out = self.decode(z)
        return out, z, mu, logvar#, self.reg_2(z)

    @staticmethod
    def _reparameterize(mu, logvar, age_disentanglement=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if age_disentanglement:
            mu_feat = mu[:,:-1]
            mu_age = mu[:, -1].view(-1, 1)
            z = mu_feat + eps * std
            z = torch.cat((z, mu_age), dim=1)
        else:
            z = mu + eps * std
        return z


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=10):
        """
        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits
        """
        super(FactorVAEDiscriminator, self).__init__()

        # Activation parameters
        self.neg_slope = 0.2
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = 1000
        # theoretically 1 with sigmoid but bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, self.hidden_units)
        self.lin2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin4 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin5 = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin6 = nn.Linear(self.hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)
        return z

    def reset_parameters(self):
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            x = layer.weight
            return nn.init.kaiming_uniform_(x, a=0.2, nonlinearity='leaky_relu')




 






