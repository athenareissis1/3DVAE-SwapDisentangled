import os
import pickle
import torch.nn
import trimesh

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
from model import Model, FactorVAEDiscriminator


class ModelManager(torch.nn.Module):
    def __init__(self, configurations, device, rendering_device=None,
                 precomputed_storage_path='precomputed'):
        super(ModelManager, self).__init__()
        self._model_params = configurations['model']
        self._optimization_params = configurations['optimization']
        self._precomputed_storage_path = precomputed_storage_path
        self._normalized_data = configurations['data']['normalize_data']
        self._age_disentanglememt = configurations['data']['age_disentanglement']
        self._swap_features = configurations['data']['swap_features']
        self._old_experiment = configurations['data']['old_experiment']
        self._model_version = configurations['data']['model_version']
        self._extra_layers = configurations['data']['extra_layers']
        self._detach_features = configurations['data']['detach_features']

        self.to_mm_const = configurations['data']['to_mm_constant']
        self.device = device
        self.template = utils.load_template(
            configurations['data']['template_path'], configurations['data']['attibute_to_remove'])

        low_res_templates, down_transforms, up_transforms = \
            self._precompute_transformations()
        meshes_all_resolutions = [self.template] + low_res_templates
        spirals_indices = self._precompute_spirals(meshes_all_resolutions)

        bs = self._optimization_params['batch_size']
        self._batch_diagonal_idx = [(bs + 1) * i for i in range(bs)]

        self._w_kl_loss = float(self._optimization_params['kl_weight'])

        self._net = Model(in_channels=self._model_params['in_channels'],
                          out_channels=self._model_params['out_channels'],
                          latent_size=self._model_params['latent_size'],
                          age_disentanglement=self._age_disentanglememt,
                          swap_features=self._swap_features,
                          batch_diagonal_idx=self._batch_diagonal_idx,
                          old_experiment=self._old_experiment,
                          spiral_indices=spirals_indices,
                          down_transform=down_transforms,
                          up_transform=up_transforms,
                          mlp_dropout=self._optimization_params['mlp_dropout'], 
                          mlp_layer_2=self._optimization_params['mlp_layer_2'],
                          mlp_layer_3=self._optimization_params['mlp_layer_3'],
                          model_version=self._model_version,
                          extra_layers=self._extra_layers,
                          detach_features=self._detach_features,
                          is_vae=self._w_kl_loss > 0).to(device)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=self._optimization_params['weight_decay'])

        self._latent_regions = self._compute_latent_regions()

        self._losses = None
        self._w_latent_cons_loss = float(
            self._optimization_params['latent_consistency_weight'])
        self._w_laplacian_loss = float(
            self._optimization_params['laplacian_weight'])
        self._w_dip_loss = float(self._optimization_params['dip_weight'])
        self._w_factor_loss = float(self._optimization_params['factor_weight'])
        self._w_age_loss = float(self._optimization_params['age_weight'])
        self._w_no_age_loss = float(self._optimization_params['no_age_weight'])

        self._rend_device = rendering_device if rendering_device else device
        self.default_shader = HardGouraudShader(
            cameras=FoVPerspectiveCameras(),
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.simple_shader = ShadelessShader(
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.renderer = self._create_renderer()

        self._swap_features = configurations['data']['swap_features']
        if self._swap_features:
            self._out_grid_size = self._optimization_params['batch_size']
        else:
            self._out_grid_size = 4

        if self._w_latent_cons_loss > 0:
            assert self._swap_features
        if self._w_dip_loss > 0:
            assert self._w_kl_loss > 0
        if self._w_factor_loss > 0:
            assert not self._swap_features
            self._factor_discriminator = FactorVAEDiscriminator(
                self._model_params['latent_size']).to(device)
            self._disc_optimizer = torch.optim.Adam(
                self._factor_discriminator.parameters(),
                lr=float(self._optimization_params['lr']), betas=(0.5, 0.9),
                weight_decay=self._optimization_params['weight_decay'])
            

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl', 'dip', 'factor',
                'latent_consistency', 'laplacian', 'age', 'no_age', 'tot']

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
                                    'transforms.pkl')
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

            with open(storage_path, 'wb') as file:
                pickle.dump(
                    [low_res_templates, down_transforms, up_transforms], file)

        down_transforms = [d.to(self.device) for d in down_transforms]
        up_transforms = [u.to(self.device) for u in up_transforms]
        return low_res_templates, down_transforms, up_transforms

    def _precompute_spirals(self, templates):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'spirals.pkl')
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
        if self._age_disentanglememt == True:
            latent_size = self._model_params['latent_size'] - 1
        else:
            latent_size = self._model_params['latent_size']
        assert latent_size % len(region_names) == 0
        region_size = latent_size // len(region_names)
        return {k: [i * region_size, (i + 1) * region_size]
                for i, k in enumerate(region_names)}

    def forward(self, data):
        return self._net(data.x)

    @torch.no_grad()
    def encode(self, data):
        self._net.eval()
        return self._net.encode(data)[0]
    
    # @torch.no_grad()
    # def mlp_features(self, mu_features):
    #     self._net.eval()
    #     return self._net.mlp_feature(mu_features)

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
            self._optimizer.zero_grad()

        data = data.to(device)

        reconstructed, z, mu, logvar, mlp_output = self.forward(data)
        
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)

        if self._w_kl_loss > 0:
            loss_kl = self._compute_kl_divergence_loss(mu, logvar, self._age_disentanglememt, self._old_experiment, self._model_version)
        else:
            loss_kl = torch.tensor(0, device=device)

        if self._w_dip_loss > 0:
            loss_dip = self._compute_dip_loss(mu, logvar)
        else:
            loss_dip = torch.tensor(0, device=device)

        if self._w_latent_cons_loss > 0 and self._swap_features:
            if self._model_version == 1: 
                # MODEL 1 & MODEL 1.1
                loss_z_cons = self._compute_latent_consistency(z, data.swapped)
            else:
                # MODEL 2 & MODEL 2.1 & MODEL 2.2 & MODEL 2.3
                loss_z_cons = self._compute_latent_consistency(mu, data.swapped)
        else:
            loss_z_cons = torch.tensor(0, device=device)
        
        if self._age_disentanglememt:
            if self._w_age_loss > 0: 
                loss_age = self._compute_age_loss(mu, data)
            else: 
                loss_age = torch.tensor(0, device=device)

            if self._w_no_age_loss > 0: 
                loss_no_age = self._compute_no_age_loss(mlp_output, data)
            else: 
                loss_no_age = torch.tensor(0, device=device)
        else:
            loss_age = torch.tensor(0, device=device)
            loss_no_age = torch.tensor(0, device=device)


        if self._extra_layers == False or self._detach_features == True:
            # MODEL 1 & MODEL 2 & MODEL 2.2
            loss_tot = loss_recon + \
                self._w_kl_loss * loss_kl + \
                self._w_dip_loss * loss_dip + \
                self._w_latent_cons_loss * loss_z_cons + \
                self._w_laplacian_loss * loss_laplacian + \
                self._w_age_loss * loss_age + \
                self._w_no_age_loss * loss_no_age

            if train:

                loss_tot.backward()
                self._optimizer.step()

         
            return {'reconstruction': loss_recon.item(),
                    'kl': loss_kl.item(),
                    'dip': loss_dip.item(),
                    'factor': 0,
                    'latent_consistency': loss_z_cons.item(),
                    'laplacian': loss_laplacian.item(),
                    'age': loss_age.item(),
                    'no_age': loss_no_age.item(),
                    'tot': loss_tot.item()}

        else:
            # MODEL 1.1 & MODEL 2.1 & MODEL 2.3
            loss_tot = loss_recon + \
                self._w_kl_loss * loss_kl + \
                self._w_dip_loss * loss_dip + \
                self._w_latent_cons_loss * loss_z_cons + \
                self._w_laplacian_loss * loss_laplacian + \
                self._w_age_loss * loss_age
            
            loss_no_age = self._w_no_age_loss * loss_no_age

            if train:
                loss_no_age.backward(retain_graph=True)

                self._net.en_layers.zero_grad()

                loss_tot.backward()
                self._optimizer.step()

            return {'reconstruction': loss_recon.item(),
                    'kl': loss_kl.item(),
                    'dip': loss_dip.item(),
                    'factor': 0,
                    'latent_consistency': loss_z_cons.item(),
                    'laplacian': loss_laplacian.item(),
                    'age': loss_age.item(),
                    'no_age': loss_no_age.item(),
                    'tot': loss_tot.item()}

    def _do_factor_vae_iteration(self, data, age, device='cpu', train=True):
        # Factor-vae split data into two batches.
        data = data.to(device)
        batch_size = data.x.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.x.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        reconstructed1, z1, mu1, logvar1 = self._net(data1)
        loss_recon = self.compute_mse_loss(reconstructed1, data1)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed1)

        loss_kl = self._compute_kl_divergence_loss(mu1, logvar1, self._age_disentanglememt, self._old_experiment, self._model_version)

        disc_z = self._factor_discriminator(z1)
        factor_loss = (disc_z[:, 0] - disc_z[:, 1]).mean()

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_laplacian_loss * loss_laplacian + \
            self._w_factor_loss * factor_loss

        if train:
            self._optimizer.zero_grad()
            loss_tot.backward(retain_graph=True)

            _, z2, _, _ = self._net(data2)
            z2_perm = self._permute_latent_dims(z2).detach()
            disc_z_perm = self._factor_discriminator(z2_perm)
            ones = torch.ones(half_batch_size, dtype=torch.long,
                              device=self.device)
            zeros = torch.zeros_like(ones)
            disc_factor_loss = 0.5 * (cross_entropy(disc_z, zeros) +
                                      cross_entropy(disc_z_perm, ones))

            self._disc_optimizer.zero_grad()
            disc_factor_loss.backward()
            self._optimizer.step()
            self._disc_optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'dip': 0,
                'factor': factor_loss.item(),
                'latent_consistency': 0,
                'laplacian': loss_laplacian.item(),
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
    def _compute_kl_divergence_loss(mu, logvar, age_disentanglememt, old_experiment, model_version):
        if age_disentanglememt and old_experiment==False and model_version!=2.3:
            mu = mu[:,:-1]
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
        latent_region = self._latent_regions[swapped_feature]
        if self._age_disentanglememt and self._old_experiment==False:
            z = z[:, :-1]
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
    
    
    def _compute_age_loss(self, mu, data):
        age = data.age.to(dtype=torch.float32)
        age_loss = self.compute_mse_loss(mu[self.batch_diagonal_idx, -1], age.squeeze())
        return age_loss
    
    def _compute_no_age_loss(self, mlp_output, data):
        age = data.age.to(dtype=torch.float32)
        no_age_loss = self.compute_mse_loss(mlp_output, age)
        return no_age_loss

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

    def log_losses(self, writer, epoch, phase='train'):
        for k in self.loss_keys:
            loss = self._losses[k]
            loss = loss.item() if torch.is_tensor(loss) else loss
            writer.add_scalar(
                phase + '/' + str(k), loss, epoch + 1)

    def log_images(self, in_data, writer, epoch, normalization_dict=None,
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
        
        cam_light_dist = 0.05

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
        torch.save({'optimizer': self._optimizer.state_dict()}, opt_name)

    def resume(self, checkpoint_dir):
        last_model_name = utils.get_model_list(checkpoint_dir, 'model')
        state_dict = torch.load(last_model_name)
        self._net.load_state_dict(state_dict['model'])
        epochs = int(last_model_name[-11:-3])
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self._optimizer.load_state_dict(state_dict['optimizer'])
        print(f"Resume from epoch {epochs}")
        return epochs


class ShadelessShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = \
            blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        pixel_colors = meshes.sample_textures(fragments)
        images = hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images
