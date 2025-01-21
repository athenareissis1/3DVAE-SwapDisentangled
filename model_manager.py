import os
import pickle
import torch.nn
import trimesh
import neptune
import numpy as np
import random

from torch.nn.functional import cross_entropy
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    BlendParams,
    HardGouraudShader
)

import utils
from mesh_simplification import MeshSimplifier
from compute_spirals import preprocess_spiral
from model import Model, AgeVAEDiscriminator, FactorVAEDiscriminator
from contrastive_loss import SNNRegLoss
from sklearn.feature_selection import mutual_info_regression
import torch.nn as nn


class ModelManager(torch.nn.Module):
    def __init__(self, configurations, device, rendering_device=None,
                 precomputed_storage_path='precomputed'):
        super(ModelManager, self).__init__()
        self._model_params = configurations['model']
        self._optimization_params = configurations['optimization']
        self._dataset_age_range = configurations['data']['dataset_age_range']
        self._swap_feature = configurations['data']['swap_features']
        self._precomputed_storage_path = precomputed_storage_path
        self._normalized_data = configurations['data']['normalize_data']
        self._age_disentanglement = configurations['model']['age_disentanglement']
        self._age_per_feature = configurations['model']['age_per_feature']
        self._contrastive_loss = configurations['model']['contrastive_loss']
        self._mi_loss = configurations['model']['mi_loss']
        self._adversarial_loss = configurations['model']['adversarial_loss']
        self._discriminator_type = configurations['model']['discriminator_type']
        self._data_type = configurations['data']['dataset_type'].split("_", 1)[1]
        self._inter_layer = configurations['model']['intermediate_layers']
        self._inter_layer_count = configurations['model']['intermediate_layers_count']
        self._inter_layer_size = configurations['model']['intermediate_layers_size']
        self._latent_size = configurations['model']['latent_size']
        self._age_latent_size = configurations['model']['age_latent_size']

        self.to_mm_const = configurations['data']['to_mm_constant']
        self.device = device
        self.template = utils.load_template(
            configurations['data']['template_path'])

        low_res_templates, down_transforms, up_transforms = \
            self._precompute_transformations()
        meshes_all_resolutions = [self.template] + low_res_templates
        spirals_indices = self._precompute_spirals(meshes_all_resolutions)

        self._w_kl_loss = float(self._optimization_params['kl_weight'])

        bs = self._optimization_params['batch_size']
        self._batch_diagonal_idx = [(bs + 1) * i for i in range(bs)]

        self._net = Model(in_channels=self._model_params['in_channels'],
                          out_channels=self._model_params['out_channels'],
                          latent_size=self._latent_size,
                          age_latent_size=self._age_latent_size,
                          inter_layer_count=self._inter_layer_count,
                          inter_layer_size=self._inter_layer_size,
                          spiral_indices=spirals_indices,
                          down_transform=down_transforms,
                          up_transform=up_transforms,
                          diagonal_idx=self._batch_diagonal_idx, 
                          is_vae=self._w_kl_loss > 0,
                          age_disentanglement=self._age_disentanglement, 
                          swap_feature=self._swap_feature,
                          inter_layer=self._inter_layer).to(device)

        self._optimizer_vae = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=self._optimization_params['weight_decay'])
        
        self._losses = None
        self._w_reconstruction_weight = float(self._optimization_params['reconstruction_weight'])
        self._w_latent_cons_loss = float(self._optimization_params['latent_consistency_weight'])
        self._w_laplacian_loss = float(self._optimization_params['laplacian_weight'])
        self._w_dip_loss = float(self._optimization_params['dip_weight'])
        self._w_factor_loss = float(self._optimization_params['factor_weight'])
        self._w_age_loss = float(self._optimization_params['age_weight'])
        self._w_contrastive_loss = float(self._optimization_params['contrastive_weight'])
        self._w_mi_loss = float(self._optimization_params['mi_weight'])
        self._w_adversarial_loss = float(self._optimization_params['adversarial_weight'])

        self._rend_device = rendering_device if rendering_device else device
        self.default_shader = HardGouraudShader(
            cameras=FoVPerspectiveCameras(),
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.simple_shader = ShadelessShader(
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.renderer = self._create_renderer()

        if self._swap_feature:
            self._latent_regions = self._compute_latent_regions()
            self._out_grid_size = self._optimization_params['batch_size']
        else:
            self._out_grid_size = 4
        if self._w_latent_cons_loss > 0:
            assert self._swap_feature
        if self._w_age_loss > 0:
            assert self._age_disentanglement
        if self._w_contrastive_loss > 0:
            assert self._contrastive_loss
        if self._w_mi_loss > 0:
            assert self._mi_loss
        if self._w_adversarial_loss > 0:
            assert self._adversarial_loss
            if self._discriminator_type == "MLP":
                self._net_discriminator = AgeVAEDiscriminator(
                    discriminator_type=self._discriminator_type,
                    input_dim = self.template.size(1) * self._model_params['in_channels'],
                    in_channels = None,
                    out_channels = None, 
                    spiral_indices = None,
                    down_transform = None).to(device)
            else:
                self._net_discriminator = AgeVAEDiscriminator(
                    discriminator_type=self._discriminator_type,
                    input_dim = None,
                    in_channels=self._model_params['in_channels'],
                    out_channels=self._model_params['out_channels'],
                    spiral_indices=spirals_indices, 
                    down_transform=down_transforms)

            self._optimizer_discriminator = torch.optim.Adam(
                self._net_discriminator.parameters(),
                lr=float(self._optimization_params['lr']),
                weight_decay=self._optimization_params['weight_decay'])
        if self._w_dip_loss > 0:
            assert self._w_kl_loss > 0
        if self._w_factor_loss > 0:
            assert not self._swap_feature
            self._factor_discriminator = FactorVAEDiscriminator(
                self._model_params['latent_size']).to(device)
            self._disc_optimizer = torch.optim.Adam(
                self._factor_discriminator.parameters(),
                lr=float(self._optimization_params['lr']), betas=(0.5, 0.9),
                weight_decay=self._optimization_params['weight_decay'])

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'dip', 'factor',
                'latent_consistency', 'laplacian', 'age', 'contrastive', 'mi', 'adversarial', 'tot']
    
    @property
    def latent_regions(self):
        return self._latent_regions

    @property
    def is_vae(self):
        return self._w_kl_loss > 0

    @property
    def model_latent_size(self):
        return self._model_params['latent_size']

    @property
    def batch_diagonal_idx(self):
        return self._batch_diagonal_idx

    def _precompute_transformations(self):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    f'transforms_{self._data_type}.pkl')
        try:
            with open(storage_path, 'rb') as file:
                low_res_templates, down_transforms, up_transforms = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Computing Down- and Up- sampling transformations ")
            if not os.path.isdir(self._precomputed_storage_path):
                os.mkdir(self._precomputed_storage_path)

            sampling_params = self._model_params['sampling']
            m = self.template

            r_weighted = False if sampling_params['type'] == 'basic' else True

            low_res_templates = []
            down_transforms = []
            up_transforms = []
            for sampling_factor in sampling_params['sampling_factors']:
                simplifier = MeshSimplifier(in_mesh=m, debug=False)
                m, down, up = simplifier(sampling_factor, r_weighted)
                low_res_templates.append(m)
                down_transforms.append(down)
                up_transforms.append(up)

            #########

            # Count vertices in each feature after downsampling
            feature_vertex_counts = {k: len(v['feature']) for k, v in m.feat_and_cont.items()}
            print(f"Vertex counts: {feature_vertex_counts}")


            # Print the vertex counts for each feature
            for feature, count in feature_vertex_counts.items():
                print(f"Feature '{feature}' has {count} vertices.")

            #########

            with open(storage_path, 'wb') as file:
                pickle.dump(
                    [low_res_templates, down_transforms, up_transforms], file)

        down_transforms = [d.to(self.device) for d in down_transforms]
        up_transforms = [u.to(self.device) for u in up_transforms]
        return low_res_templates, down_transforms, up_transforms

    def _precompute_spirals(self, templates):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    f'spirals_{self._data_type}.pkl')
        try:
            with open(storage_path, 'rb') as file:
                spiral_indices_list = pickle.load(file)
        except FileNotFoundError:
            print("Computing Spirals")
            spirals_params = self._model_params['spirals']
            spiral_indices_list = []
            for i in range(len(templates) - 1):
                spiral_indices_list.append(
                    preprocess_spiral(templates[i].face.t().cpu().numpy(),
                                      spirals_params['length'][i],
                                      templates[i].pos.cpu().numpy(),
                                      spirals_params['dilation'][i]))
            with open(storage_path, 'wb') as file:
                pickle.dump(spiral_indices_list, file)
        spiral_indices_list = [s.to(self.device) for s in spiral_indices_list]
        return spiral_indices_list

    def _compute_latent_regions(self):

        region_names = list(self.template.feat_and_cont.keys())
        latent_size = self._model_params['latent_size'] # not including age latents 

        if self._model_params['age_disentanglement'] and self._model_params['age_per_feature'] == False:
            latent_size -= self._model_params['age_latent_size']

        assert latent_size % len(region_names) == 0
        region_size = latent_size // len(region_names)
        return {k: [i * region_size, (i + 1) * region_size]
                for i, k in enumerate(region_names)}

    def _gt_age(self, bs, age_latents, gt_age, swapped_feature):

        if self._swap_feature:
            if self._model_params['age_disentanglement'] and self._model_params['age_per_feature'] == False:
                latent_per_feature = (self._latent_size - self._age_latent_size) // self._age_latent_size
            else:
                latent_per_feature = self._latent_size // self._age_latent_size
            swapped_latent_index = self._latent_regions[swapped_feature][0] // latent_per_feature
            # make a gt_age matrix of size [16,9]
            gt_feature_ages = torch.zeros([bs ** 2, self._age_latent_size],
                    device=age_latents.device,
                    dtype=age_latents.dtype)

            # make new gt_age matrix with swapped feature ages
            for j in range(bs):
                for i in range(bs):
                    gt_feature_ages[i * bs + j, ::] = gt_age[i, ::]
                    if i != j:
                        gt_feature_ages[i * bs + j, swapped_latent_index] = gt_age[j]
                
        else:
            gt_age = gt_age.repeat(1, 9)
            gt_feature_ages = torch.zeros([bs, self._age_latent_size],
                                            device=age_latents.device,
                                            dtype=age_latents.dtype)
            for i in range(bs):
                gt_feature_ages[i, ::] = gt_age[i, ::]

        return gt_feature_ages

    def forward(self, data):
        return self._net(data)

    @torch.no_grad()
    def encode(self, data):
        self._net.eval()
        return self._net.encode(data)[0]

    @torch.no_grad()
    def generate(self, z):
        self._net.eval()
        return self._net.decode(z)

    def generate_for_opt(self, z):
        self._net.train()
        return self._net.decode(z)

    def run_epoch(self, data_loader, device, train=True):
        if train:
            self._net.train()
        else:
            self._net.eval()

        if self._w_factor_loss > 0:
            iteration_function = self._do_factor_vae_iteration
        else:
            iteration_function = self._do_iteration

        self._reset_losses()
        it = 0
        for it, data in enumerate(data_loader):
            if train:
                losses = iteration_function(data, device, train=True)
            else:
                with torch.no_grad():
                    losses = iteration_function(data, device, train=False)
            self._add_losses(losses)
        self._divide_losses(it + 1)

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer_vae.zero_grad()
                
        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data.x)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)

        if self._w_kl_loss > 0:
            loss_kl = self._compute_kl_divergence_loss(mu, logvar, self._age_disentanglement, self._age_latent_size)
        else:
            loss_kl = torch.tensor(0, device=device)

        if self._w_dip_loss > 0:
            loss_dip = self._compute_dip_loss(mu, logvar)
        else:
            loss_dip = torch.tensor(0, device=device)

        if self._swap_feature:
            swapped = data.swapped 
            loss_z_cons = self._compute_latent_consistency(mu, swapped)
        else:
            swapped = None 
            loss_z_cons = torch.tensor(0, device=device)

        if self._age_disentanglement:
            loss_age = self._compute_age_loss(mu, data.norm_age, swapped)
        else:
            loss_age = torch.tensor(0, device=device)

        if self._contrastive_loss:
            loss_contrastive = self._compute_contrastive_loss(mu, data.norm_age, swapped)
        else:
            loss_contrastive = torch.tensor(0, device=device)

        if self._mi_loss:
            loss_mi = self._compute_mi_loss(mu)
        else:
            loss_mi = torch.tensor(0, device=device)

        if self._adversarial_loss and train:
            loss_adversarial, discriminator_loss = self._compute_and_train_adversarial_loss(data.x, mu, reconstructed, data.norm_age)
        else:
            loss_adversarial = torch.tensor(0, device=device)
            discriminator_loss = torch.tensor(0, device=device)

        loss_tot = self._w_reconstruction_weight * loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_dip_loss * loss_dip + \
            self._w_latent_cons_loss * loss_z_cons + \
            self._w_laplacian_loss * loss_laplacian + \
            self._w_age_loss * loss_age + \
            self._w_contrastive_loss * loss_contrastive + \
            self._w_mi_loss * loss_mi + \
            self._w_adversarial_loss * loss_adversarial

        if train:
            if self._adversarial_loss:
                loss_tot.backward(retain_graph=True)
                self._optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                self._optimizer_vae.step() 
                self._optimizer_discriminator.step()
            else:
                loss_tot.backward()
                self._optimizer_vae.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'dip': loss_dip.item(),
                'factor': 0,
                'latent_consistency': loss_z_cons.item(),
                'laplacian': loss_laplacian.item(),
                'age': loss_age.item(),
                'contrastive': loss_contrastive.item(),
                'mi': loss_mi.item(),
                'adversarial': loss_adversarial.item(),
                'discriminator': discriminator_loss,
                'tot': loss_tot.item()}

    def _do_factor_vae_iteration(self, data, device='cpu', train=True):
        # Factor-vae split data into two batches.
        data = data.to(device)
        batch_size = data.x.size(dim=0)
        half_batch_size = batch_size // 2
        data_x = data.x.split(half_batch_size)
        data_x_1 = data_x[0]
        data_x_2 = data_x[1]

        # Factor VAE Loss
        reconstructed1, z1, mu1, logvar1 = self._net(data_x_1)
        loss_recon = self.compute_mse_loss(reconstructed1, data_x_1)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed1)

        loss_kl = self._compute_kl_divergence_loss(mu1, logvar1, self._age_disentanglement, self._age_latent_size)

        if self._age_disentanglement:
            norm_age = data.norm_age.split(half_batch_size)[0]
            if self._swap_feature:
                swapped = data.swapped.split(half_batch_size)[0]
            else:
                swapped = None
            loss_age = self._compute_age_loss(mu1, norm_age, swapped)
        else:
            loss_age = torch.tensor(0, device=device)

        disc_z = self._factor_discriminator(z1)
        factor_loss = (disc_z[:, 0] - disc_z[:, 1]).mean()

        loss_tot = self._w_reconstruction_weight * loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_laplacian_loss * loss_laplacian + \
            self._w_age_loss * loss_age + \
            self._w_factor_loss * factor_loss 

        if train:
            self._optimizer_vae.zero_grad()
            loss_tot.backward(retain_graph=True)

            _, z2, _, _ = self._net(data_x_2)
            z2_perm = self._permute_latent_dims(z2).detach()
            disc_z_perm = self._factor_discriminator(z2_perm)
            ones = torch.ones(half_batch_size, dtype=torch.long,
                              device=self.device)
            zeros = torch.zeros_like(ones)
            disc_factor_loss = 0.5 * (cross_entropy(disc_z, zeros) +
                                      cross_entropy(disc_z_perm, ones))

            self._disc_optimizer.zero_grad()
            disc_factor_loss.backward()
            self._optimizer_vae.step()
            self._disc_optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'dip': 0,
                'factor': factor_loss.item(),
                'latent_consistency': 0,
                'laplacian': loss_laplacian.item(),
                'age': loss_age.item(),
                'contrastive': 0,
                'tot': loss_tot.item()}
      

    @staticmethod
    def _compute_l1_loss(prediction, gt, reduction='mean'):
        return torch.nn.L1Loss(reduction=reduction)(prediction, gt)

    @staticmethod
    def compute_mse_loss(prediction, gt, reduction='mean'):
        return torch.nn.MSELoss(reduction=reduction)(prediction, gt)

    def _compute_laplacian_regularizer(self, prediction):
        bs = prediction.shape[0]
        n_verts = prediction.shape[1]
        laplacian = self.template.laplacian.to(prediction.device)
        prediction_laplacian = utils.batch_mm(laplacian, prediction)
        loss = prediction_laplacian.norm(dim=-1) / n_verts
        return loss.sum() / bs

    @staticmethod
    def _compute_kl_divergence_loss(mu, logvar, age_disentanglement, age_latent_size):
        if age_disentanglement:
            mu = mu[:,:-age_latent_size]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl, dim=0)
            
    def _compute_dip_loss(self, mu, logvar):
        centered_mu = mu - mu.mean(dim=1, keepdim=True)
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze()

        if self._optimization_params['dip_type'] == 'ii':
            cov_z = cov_mu + torch.mean(
                torch.diagonal((2. * logvar).exp(), dim1=0), dim=0)
        else:
            cov_z = cov_mu

        cov_diag = torch.diag(cov_z)
        cov_offdiag = cov_z - torch.diag(cov_diag)

        lambda_diag = self._optimization_params['dip_diag_lambda']
        lambda_offdiag = self._optimization_params['dip_offdiag_lambda']
        return lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
            lambda_diag * torch.sum((cov_diag - 1) ** 2)

    def _compute_latent_consistency(self, z, swapped_feature):
        bs = self._optimization_params['batch_size']
        eta1 = self._optimization_params['latent_consistency_eta1']
        eta2 = self._optimization_params['latent_consistency_eta2']
        # latent_region = self._latent_regions_age[swapped_feature]
        latent_region = self._latent_regions[swapped_feature]

        if self._age_disentanglement:
            #  OLD IMPLEMENTATION - do not include the single age latent in LC loss
            if self._model_params['age_latent_size'] == 1:
                z = z[:, :-self._age_latent_size]
            #  NEW IMPLEMENTATION - include all age latents in LC loss
            else:
                index_list = []
                special_index = self.model_latent_size - self._age_latent_size

                for i in range(special_index):
                    index_list.append(i)
                    if (i + 1) % 5 == 0:
                        index_list.append(special_index)
                        special_index += 1

                # indexes on new order of latents to have the feature age latents at the end of the feature shape latents
                index_tensor = torch.tensor(index_list, dtype=torch.long)

                z = z[:, index_tensor]

        z_feature = z[:, latent_region[0]:latent_region[1]].view(bs, bs, -1)
        z_else = torch.cat([z[:, :latent_region[0]],
                            z[:, latent_region[1]:]], dim=1).view(bs, bs, -1)

        triu_indices = torch.triu_indices(
            z_feature.shape[0], z_feature.shape[0], 1)

        lg = z_feature.unsqueeze(0) - z_feature.unsqueeze(1)
        lg = lg[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                lg.shape[-1])
        lg = torch.sum(lg ** 2, dim=-1)

        dg = z_feature.permute(1, 2, 0).unsqueeze(0) - \
            z_feature.permute(1, 2, 0).unsqueeze(1)
        dg = dg[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        dg = torch.sum(dg.reshape(-1, dg.shape[-1]) ** 2, dim=-1)

        dr = z_else.unsqueeze(0) - z_else.unsqueeze(1)
        dr = dr[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                dr.shape[-1])
        dr = torch.sum(dr ** 2, dim=-1)

        lr = z_else.permute(1, 2, 0).unsqueeze(0) - \
            z_else.permute(1, 2, 0).unsqueeze(1)
        lr = lr[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        lr = torch.sum(lr.reshape(-1, lr.shape[-1]) ** 2, dim=-1)
        zero = torch.tensor(0, device=z.device)
        return (1 / (bs ** 3 - bs ** 2)) * \
               (torch.sum(torch.max(zero, lr - dr + eta2)) +
                torch.sum(torch.max(zero, lg - dg + eta1)))
    
    def _compute_age_loss(self, mu, gt_age, swapped_feature):

        age_latents = mu[:, -self._age_latent_size:]

        if self._age_per_feature: 
            assert self._age_latent_size > 1
            
            bs = self._optimization_params['batch_size']
            # bs = mu.shape[0]

            gt_feature_ages = self._gt_age(bs, age_latents, gt_age, swapped_feature)
            
            age_loss = self.compute_mse_loss(age_latents, gt_feature_ages)

        else:
            if self._swap_feature:
                age_latents = age_latents[self.batch_diagonal_idx]

            gt_age = gt_age.to(dtype=torch.float32).squeeze()
            age_loss = self.compute_mse_loss(age_latents, gt_age)

        return age_loss
    
    def _compute_contrastive_loss(self, latent, gt_age, swapped_feature):
        # add in if age_per_feature
        # if self._swap_feature:
        #     latent = latent[self.batch_diagonal_idx]

        age_latents = latent[:, -self._age_latent_size:]
        bs = self._optimization_params['batch_size']
        gt_ages = self._gt_age(bs, age_latents, gt_age, swapped_feature)

        SNN_Loss_Reg = SNNRegLoss(self._optimization_params['contrastive_loss_temp'], self._optimization_params['contrastive_loss_threshold'])
        # SNN_Loss_Reg = SNNRegLoss(self._optimization_params['contrastive_loss_temp'], self._optimization_params['contrastive_loss_threshold'], self._model_params['age_latent_size'])
        loss_snn_reg = SNN_Loss_Reg(latent, gt_ages)

        return loss_snn_reg

    def _compute_mi_loss(self, mu):

        age_latents_size = self._model_params['age_latent_size']

        identity_latents = mu[:, :-self._age_latent_size].detach().cpu().numpy()
        age_latents = mu[:, -self._age_latent_size:].detach().cpu().numpy()

        if age_latents_size == 1:
            mi_loss = mutual_info_regression(identity_latents, age_latents.ravel())
        else:
            mi_matrix = np.zeros((identity_latents.shape[1], age_latents.shape[1]))

            for i in range(identity_latents.shape[1]): 
                for j in range(age_latents.shape[1]):  
                        mi_matrix[i, j] = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents[:, j]).mean()

            mi_loss = np.mean(mi_matrix)

        return mi_loss
    
    def reconstructed_new_ages(self, z, gt_ages):

    ### 3 NEW AGES IMPLEMENTATION ###

        bs = gt_ages.size(0)
        age_size = self._age_latent_size

        decrease_factor = 0.2 
        increase_factor = 0.1 

        age_min, age_max = map(int, self._dataset_age_range.split('-'))

        storage_path = os.path.join(self._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        try:
            with open(storage_path, 'rb') as file:
                age_mean, age_std = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Could not find normalise stats file")

        # single value
        age_min_norm = (age_min - age_mean) / age_std
        age_max_norm = (age_max - age_mean) / age_std
        
        age_min_tensor = torch.tensor(age_min_norm, dtype=gt_ages.dtype, device=gt_ages.device)
        age_max_tensor = torch.tensor(age_max_norm, dtype=gt_ages.dtype, device=gt_ages.device)

        decreased_age = gt_ages - (gt_ages * decrease_factor)
        increased_age = gt_ages + (gt_ages * increase_factor)

        output_decreased_ages = torch.where(decreased_age > age_min_tensor, decreased_age, gt_ages + (gt_ages * (2 * increase_factor)))
        output_increased_ages = torch.where(increased_age < age_max_tensor, increased_age, gt_ages - (gt_ages * (2 * decrease_factor)))

        while True:
            random_ages = torch.randint(age_min, age_max + 1, (bs,1), dtype=gt_ages.dtype, device=gt_ages.device)
            random_ages_norm = (random_ages - age_mean) / age_std
            if not torch.equal(random_ages_norm, gt_ages):
                break

        output_decreased_ages_expanded = output_decreased_ages.expand(-1, age_size)
        output_increased_ages_expanded = output_increased_ages.expand(-1, age_size)
        random_ages_norm_expanded = random_ages_norm.expand(-1, age_size)
        
        new_ages = torch.cat((output_decreased_ages_expanded, output_increased_ages_expanded, random_ages_norm_expanded), dim=0)
        z_expanded = z.repeat(bs-1, 1)

        z_expanded[:, -age_size:] = new_ages
        reconstructed = self.generate(z_expanded) 

        return reconstructed

    #### ONE NEW AGE IMPLEMENTATION ####

        # bs = gt_ages.size(0)
        # age_size = self._age_latent_size

        # age_min, age_max = map(int, self._dataset_age_range.split('-'))

        # storage_path = os.path.join(self._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        # try:
        #     with open(storage_path, 'rb') as file:
        #         age_mean, age_std = \
        #             pickle.load(file)
        # except FileNotFoundError:
        #     print("Could not find normalise stats file")

        # while True:
        #     random_ages = torch.randint(age_min, age_max + 1, (bs,1), dtype=gt_ages.dtype, device=gt_ages.device)
        #     random_ages_norm = (random_ages - age_mean) / age_std
        #     if not torch.equal(random_ages_norm, gt_ages):
        #         break

        # random_ages_norm_expanded = random_ages_norm.expand(-1, age_size) 

        # new_ages = random_ages_norm_expanded
        # z[:, -age_size:] = new_ages
        # reconstructed = self.generate(z) 

        # return reconstructed

    def _compute_and_train_adversarial_loss(self, real_data, z, reconstructed_data, gt_ages):
        bs_fake_data = 4  #### change when using 1 new age or 3 new ages 

        real_data = real_data[self.batch_diagonal_idx]
        z = z[self.batch_diagonal_idx]
        reconstructed_data = reconstructed_data[self.batch_diagonal_idx]

        reconstructed_data_new_ages = self.reconstructed_new_ages(z, gt_ages)
        real_data = real_data.repeat(bs_fake_data, 1, 1)
        reconstructed_data = torch.cat((reconstructed_data, reconstructed_data_new_ages), dim=0)

        bs = real_data.size(0)

        real_labels = torch.ones(bs, 1).to(self.device)
        fake_labels = torch.zeros(bs, 1).to(self.device)

        if self._discriminator_type == "MLP":
            real_data = real_data.view(bs, -1)
            reconstructed_data = reconstructed_data.view(bs, -1)

        real_output = self._net_discriminator(real_data)
        fake_output = self._net_discriminator(reconstructed_data.detach())

        d_loss_real = nn.BCELoss()(real_output, real_labels)
        d_loss_fake = nn.BCELoss()(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        g_loss = nn.BCELoss()(fake_output, real_labels)

        return g_loss, d_loss

    @staticmethod
    def _permute_latent_dims(latent_sample):
        perm = torch.zeros_like(latent_sample)
        batch_size, dim_z = perm.size()
        for z in range(dim_z):
            pi = torch.randperm(batch_size).to(latent_sample.device)
            perm[:, z] = latent_sample[pi, z]
        return perm

    def compute_vertex_errors(self, out_verts, gt_verts):
        vertex_errors = self.compute_mse_loss(
            out_verts, gt_verts, reduction='none')
        vertex_errors = torch.sqrt(torch.sum(vertex_errors, dim=-1))
        vertex_errors *= self.to_mm_const
        return vertex_errors

    def _reset_losses(self):
        self._losses = {k: 0 for k in self.loss_keys}

    def _add_losses(self, additive_losses):
        for k in self.loss_keys:
            loss = additive_losses[k]
            self._losses[k] += loss.item() if torch.is_tensor(loss) else loss

    def _divide_losses(self, value):
        for k in self.loss_keys:
            self._losses[k] /= value

    def log_losses(self, writer, nept_log, epoch, phase='train'):
        for k in self.loss_keys:
            loss = self._losses[k]
            loss = loss.item() if torch.is_tensor(loss) else loss
            writer.add_scalar(
                phase + '/' + str(k), loss, epoch + 1)
            nept_log[phase + '/' + str(k)].log((loss))

    def log_images(self, in_data, writer, nept_log, epoch, normalization_dict=None,
                   phase='train', error_max_scale=5):
        gt_meshes = in_data.x.to(self._rend_device)
        out_meshes = self.forward(in_data.to(self.device))[0]
        out_meshes = out_meshes.to(self._rend_device)

        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            gt_meshes = gt_meshes * std_mesh + mean_mesh
            out_meshes = out_meshes * std_mesh + mean_mesh

        vertex_errors = self.compute_vertex_errors(out_meshes, gt_meshes)

        gt_renders = self.render(gt_meshes)
        out_renders = self.render(out_meshes)
        errors_renders = self.render(out_meshes, vertex_errors,
                                     error_max_scale)
        log = torch.cat([gt_renders, out_renders, errors_renders], dim=-1)
        log = make_grid(log, padding=10, pad_value=1, nrow=self._out_grid_size)
        writer.add_image(tag=phase, global_step=epoch + 1, img_tensor=log)

        img = ToPILImage()(log.cpu())
        img_np = np.array(img)

        nept_log[phase + '/images'].log(neptune.types.File.as_image(img_np))

    def _create_renderer(self, img_size=256):
        raster_settings = RasterizationSettings(image_size=img_size)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings,
                                      cameras=FoVPerspectiveCameras()),
            shader=self.default_shader)
        renderer.to(self._rend_device)
        return renderer

    @torch.no_grad()
    def render(self, batched_data, vertex_errors=None, error_max_scale=None):
        batch_size = batched_data.shape[0]
        batched_verts = batched_data.detach().to(self._rend_device)
        template = self.template.to(self._rend_device)

        if vertex_errors is not None:
            self.renderer.shader = self.simple_shader
            textures = TexturesVertex(utils.errors_to_colors(
                vertex_errors, min_value=0,
                max_value=error_max_scale, cmap='plasma') / 255)
        else:
            self.renderer.shader = self.default_shader
            textures = TexturesVertex(torch.ones_like(batched_verts) * 0.5)

        meshes = Meshes(
            verts=batched_verts,
            faces=template.face.t().expand(batch_size, -1, -1),
            textures=textures)
        
        if "lyhm" in self._data_type:
            cam_light_dist = 0.05
        else: 
            cam_light_dist = 2.4

        rotation, translation = look_at_view_transform(
            dist=cam_light_dist, elev=0, azim=15)
        cameras = FoVPerspectiveCameras(R=rotation, T=translation,
                                        device=self._rend_device, znear=0.05)

        lights = PointLights(location=[[0.0, 0.0, cam_light_dist]],
                             diffuse_color=[[1., 1., 1.]],
                             device=self._rend_device)

        materials = Materials(shininess=0.5, device=self._rend_device)

        images = self.renderer(meshes, cameras=cameras, lights=lights,
                               materials=materials).permute(0, 3, 1, 2)
        return images[:, :3, ::]

    def render_and_show_batch(self, data, normalization_dict):
        verts = data.x.to(self._rend_device)
        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            verts = verts * std_mesh + mean_mesh
        rend = self.render(verts)
        grid = make_grid(rend, padding=10, pad_value=1,
                         nrow=self._out_grid_size)
        img = ToPILImage()(grid)
        img.show()

    def show_mesh(self, vertices, normalization_dict=None):
        vertices = torch.squeeze(vertices)
        if self._normalized_data:
            mean_verts = normalization_dict['mean'].to(vertices.device)
            std_verts = normalization_dict['std'].to(vertices.device)
            vertices = vertices * std_verts + mean_verts
        mesh = trimesh.Trimesh(vertices.cpu().detach().numpy(),
                               self.template.face.t().cpu().numpy())
        mesh.show()

    def save_weights(self, checkpoint_dir, epoch):
        net_name = os.path.join(checkpoint_dir, 'model_%08d.pt' % (epoch + 1))
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save({'model': self._net.state_dict()}, net_name)
        torch.save({'optimizer': self._optimizer_vae.state_dict()}, opt_name)

    def resume(self, checkpoint_dir):
        last_model_name = utils.get_model_list(checkpoint_dir, 'model')
        state_dict = torch.load(last_model_name)
        self._net.load_state_dict(state_dict['model'])
        epochs = int(last_model_name[-11:-3])
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self._optimizer_vae.load_state_dict(state_dict['optimizer'])
        print(f"Resume from epoch {epochs}")
        return epochs
    
    def log_hyperparameters(self, log, config, logging):
        if config['model']['age_disentanglement']:
            latent_size = config['model']['latent_size'] - config['model']['age_latent_size']
        else:
            latent_size = config['model']['latent_size']

        hyperparameters = logging['logging']['neptune_logging_hyperparameters']

        for hyperparameter in hyperparameters:
            group, variable = hyperparameter.split(':')
            if variable == 'latent_size':
                value = latent_size
            else:
                value = config[group][variable]
            log[variable] = value

class ShadelessShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = \
            blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        pixel_colors = meshes.sample_textures(fragments)
        images = hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images
