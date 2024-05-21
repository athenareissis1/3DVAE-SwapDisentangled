import os
import json
import pickle
import tqdm
import trimesh
import torch.nn
import pytorch3d.loss
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets


class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, val_load, test_load, out_dir, config):
    
        self._manager = model_manager
        self._manager.eval()
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._normalized_data = config['data']['normalize_data']
        self._out_dir = out_dir
        self._config = config
        self._train_loader = train_load
        self._val_loader = val_load
        self._test_loader = test_load
        self._is_vae = self._manager.is_vae
        self.latent_stats = self.compute_latent_stats(train_load)
        self._data_type = config['data']['dataset_type'].split("_", 1)[1]

        self.coma_landmarks = [
            1337, 1344, 1163, 878, 3632, 2496, 2428, 2291, 2747,
            3564, 1611, 2715, 3541, 1576, 3503, 3400, 3568, 1519,
            203, 183, 870, 900, 867, 3536]
        self.uhm_landmarks = [
            10754, 10826, 9123, 10667, 19674, 28739, 4831, 19585,
            8003, 22260, 12492, 27386, 1969, 31925, 31158, 20963,
            1255, 9881, 32055, 45778, 5355, 27515, 18482, 33691]

    def __call__(self):
        # self.set_renderings_size(512)
        # self.set_rendering_background_color([1, 1, 1])

        # # Qualitative evaluations
        # if self._config['data']['swap_features']:
        #     self.latent_swapping(next(iter(self._test_loader)).x)
        # self.per_variable_range_experiments(use_z_stats=False)
        # self.random_generation_and_rendering(n_samples=16)
        # self.random_generation_and_save(n_samples=16)
        # self.interpolate() # not working
        # if self._config['data']['dataset_type'] == 'faces':
        #     self.direct_manipulation()

        # # Quantitative evaluation
        # self.evaluate_gen(self._test_loader, n_sampled_points=2048) # takes a while to run
        # recon_errors = self.reconstruction_errors(self._test_loader)
        # train_set_diversity = self.compute_diversity_train_set()
        # diversity = self.compute_diversity()
        # specificity = self.compute_specificity() # takes a while to run
        # metrics = {'recon_errors': recon_errors,
        #            'train_set_diversity': train_set_diversity,
        #            'diversity': diversity,
        #            'specificity': specificity}

        # outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        # with open(outfile_path, 'w') as outfile:
        #     json.dump(metrics, outfile)

        # TEST TO RUN (run all on val set then once model is finalised move to test set)
        self.set_renderings_size(512)
        self.set_rendering_background_color([1, 1, 1])
        self.per_variable_range_experiments(use_z_stats=False)
        self.random_generation_and_rendering(n_samples=16)

        if self._config['model']['age_disentanglement']:
            self.age_encoder_check(self._train_loader, self._test_loader)
            self.dataset_age_split(self._train_loader, self._val_loader, self._test_loader)
            self.age_prediction_MLP(self._train_loader, self._test_loader)

            if self._config['model']['conditional_vae']:
                self.conditional_age_latent_changing(self._val_loader)
            else:
                self.age_latent_changing(self._val_loader)
                self.age_prediction_encode_output(self._train_loader, self._val_loader)
            

    def conditional_data(self, data):

        assert self._config['data']['swap_features'] == False

        x = data.x
        age = data.norm_age

        # Expand and repeat ages to match the face_meshes shape
        age_expanded = age.unsqueeze(-1).expand(-1, x.shape[1], 1)

        # Concatenate along the last dimension
        mesh_with_age = torch.cat((x, age_expanded), dim=2)
        x = mesh_with_age.float()

        data.x = x

        return data

    def _unnormalize_verts(self, verts, dev=None):
        d = self._device if dev is None else dev
        return verts * self._norm_dict['std'].to(d) + \
            self._norm_dict['mean'].to(d)

    def set_renderings_size(self, size):
        self._manager.renderer.rasterizer.raster_settings.image_size = size

    def set_rendering_background_color(self, color=None):
        color = [1, 1, 1] if color is None else color
        blend_params = BlendParams(background_color=color)
        self._manager.default_shader.blend_params = blend_params
        self._manager.simple_shader.blend_params = blend_params

    def compute_latent_stats(self, data_loader):
        storage_path = os.path.join(self._out_dir, 'z_stats.pkl')
        try:
            with open(storage_path, 'rb') as file:
                z_stats = pickle.load(file)
        except FileNotFoundError:
            latents_list = []
            for data in tqdm.tqdm(data_loader):

                if self._config['model']['conditional_vae']:
                    data = self.conditional_data(data)
                latents_list.append(self._manager.encode(
                    data.x.to(self._device)).detach().cpu())
            latents = torch.cat(latents_list, dim=0)
            z_means = torch.mean(latents, dim=0)
            z_stds = torch.std(latents, dim=0)
            z_mins, _ = torch.min(latents, dim=0)
            z_maxs, _ = torch.max(latents, dim=0)
            z_stats = {'means': z_means, 'stds': z_stds,
                       'mins': z_mins, 'maxs': z_maxs}

            with open(storage_path, 'wb') as file:
                pickle.dump(z_stats, file)
        return z_stats

    @staticmethod
    def string_to_color(rgba_string, swap_bw=True):
        rgba_string = rgba_string[1:-1]  # remove [ and ]
        rgb_values = rgba_string.split()[:-1]
        colors = [int(c) / 255 for c in rgb_values]
        if colors == [1., 1., 1.] and swap_bw:
            colors = [0., 0., 0.]
        return tuple(colors)

    def per_variable_range_experiments(self, z_range_multiplier=1,
                                       use_z_stats=True):
        if self._is_vae and not use_z_stats and not self._config['model']['age_disentanglement']:
            latent_size = self._manager.model_latent_size
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
        elif self._is_vae and not use_z_stats and self._config['model']['age_disentanglement']:
            latent_size = self._manager.model_latent_size   
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
            z_mins[-1] = self.latent_stats['mins'][-1] * z_range_multiplier
            z_maxs[-1] = self.latent_stats['maxs'][-1] * z_range_multiplier
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

        # Create video perturbing each latent variable from min to max.
        # Show generated mesh and error map next to each other
        # Frames are all concatenated along the same direction. A black frame is
        # added before start perturbing the next latent variable
        n_steps = 10
        all_frames, all_rendered_differences, max_distances = [], [], []
        all_renderings = []
        for i in tqdm.tqdm(range(z_means.shape[0])):
            z = z_means.repeat(n_steps, 1)
            z[:, i] = torch.linspace(
                z_mins[i], z_maxs[i], n_steps).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            max_distances.append(differences_from_first[-1, ::])
            renderings = self._manager.render(gen_verts).detach().cpu()
            all_renderings.append(renderings)
            differences_renderings = self._manager.render(
                gen_verts, differences_from_first,
                error_max_scale=5).cpu().detach()
            all_rendered_differences.append(differences_renderings)
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames.append(
                torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

        write_video(
            os.path.join(self._out_dir, 'latent_exploration.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Same video as before, but effects of perturbing each latent variables
        # are shown in the same frame. Only error maps are shown.
        grid_frames = []
        grid_nrows = 8
        if self._config['data']['swap_features']:
            z_size = self._config['model']['latent_size']
            grid_nrows = z_size // len(self._manager.latent_regions)

        stacked_frames = torch.stack(all_rendered_differences)
        for i in range(stacked_frames.shape[1]):
            grid_frames.append(
                make_grid(stacked_frames[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        save_image(grid_frames[-1],
                   os.path.join(self._out_dir, 'latent_exploration_tiled.png'))
        write_video(
            os.path.join(self._out_dir, 'latent_exploration_tiled.mp4'),
            torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)

        # Same as before, but only output meshes are used
        stacked_frames_meshes = torch.stack(all_renderings)
        grid_frames_m = []
        for i in range(stacked_frames_meshes.shape[1]):
            grid_frames_m.append(
                make_grid(stacked_frames_meshes[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        write_video(
            os.path.join(self._out_dir, 'latent_exploration_outs_tiled.mp4'),
            torch.stack(grid_frames_m, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Create a plot showing the effects of perturbing latent variables in
        # each region of the face
        df = pd.DataFrame(columns=['mean_dist', 'z_var', 'region'])
        df_row = 0
        for zi, vert_distances in enumerate(max_distances):
            for region, indices in self._manager.template.feat_and_cont.items():
                regional_distances = vert_distances[indices['feature']]
                mean_regional_distance = torch.mean(regional_distances)
                df.loc[df_row] = [mean_regional_distance.item(), zi, region]
                df_row += 1

        sns.set_theme(style="ticks")
        palette = {k: self.string_to_color(k) for k in
                   self._manager.template.feat_and_cont.keys()}
        grid = sns.FacetGrid(df, col="region", hue="region", palette=palette,
                             col_wrap=4, height=3)

        grid.map(plt.plot, "z_var", "mean_dist", marker="o")
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration_split.svg'))

        sns.relplot(data=df, kind="line", x="z_var", y="mean_dist",
                    hue="region", palette=palette)
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration.svg'))

    def random_latent(self, n_samples, z_range_multiplier=1):
        if self._is_vae:  # sample from normal distribution if vae
            z = torch.randn([n_samples, self._manager.model_latent_size])
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

            uniform = torch.rand([n_samples, z_means.shape[0]],
                                 device=z_means.device)
            z = uniform * (z_maxs - z_mins) + z_mins
        return z

    def random_generation(self, n_samples=16, z_range_multiplier=1,
                          denormalize=True):
        z = self.random_latent(n_samples, z_range_multiplier)
        gen_verts = self._manager.generate(z.to(self._device))
        if self._normalized_data and denormalize:
            gen_verts = self._unnormalize_verts(gen_verts)
        return gen_verts

    def random_generation_and_rendering(self, n_samples=16,
                                        z_range_multiplier=1):
        gen_verts = self.random_generation(n_samples, z_range_multiplier)
        renderings = self._manager.render(gen_verts).cpu()
        grid = make_grid(renderings, padding=10, pad_value=1)
        save_image(grid, os.path.join(self._out_dir, 'random_generation.png'))

    def random_generation_and_save(self, n_samples=16, z_range_multiplier=1):
        out_mesh_dir = os.path.join(self._out_dir, 'random_meshes')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        gen_verts = self.random_generation(n_samples, z_range_multiplier)

        self.save_batch(gen_verts, out_mesh_dir)

    def save_batch(self, batch_verts, out_mesh_dir):
        for i in range(batch_verts.shape[0]):
            mesh = trimesh.Trimesh(
                batch_verts[i, ::].cpu().detach().numpy(),
                self._manager.template.face.t().cpu().numpy())
            mesh.export(os.path.join(out_mesh_dir, str(i) + '.ply'))

    def reconstruction_errors(self, data_loader):
        print('Compute reconstruction errors')
        data_errors = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            gt = data.x

            recon = self._manager.forward(data)[0]

            if self._normalized_data:
                gt = self._unnormalize_verts(gt)
                recon = self._unnormalize_verts(recon)

            errors = self._manager.compute_vertex_errors(recon, gt)
            data_errors.append(torch.mean(errors.detach(), dim=1))
        data_errors = torch.cat(data_errors, dim=0)
        return {'mean': torch.mean(data_errors).item(),
                'median': torch.median(data_errors).item(),
                'max': torch.max(data_errors).item()}

    def compute_diversity_train_set(self):
        print('Computing train set diversity')
        previous_verts_batch = None
        mean_distances = []
        for data in tqdm.tqdm(self._train_loader):
            if self._config['data']['swap_features']:
                x = data.x[self._manager.batch_diagonal_idx, ::]
            else:
                x = data.x

            current_verts_batch = x
            if self._normalized_data:
                current_verts_batch = self._unnormalize_verts(
                    current_verts_batch, x.device)

            if previous_verts_batch is not None:
                verts_batch_distances = self._manager.compute_vertex_errors(
                    previous_verts_batch, current_verts_batch)
                mean_distances.append(torch.mean(verts_batch_distances, dim=1))
            previous_verts_batch = current_verts_batch
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_diversity(self, n_samples=10000):
        print('Computing generative model diversity')
        samples_per_batch = 20
        mean_distances = []
        for _ in tqdm.tqdm(range(n_samples // samples_per_batch)):
            verts_batch_distances = self._manager.compute_vertex_errors(
                self.random_generation(samples_per_batch),
                self.random_generation(samples_per_batch))
            mean_distances.append(torch.mean(verts_batch_distances, dim=1))
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_specificity(self, n_samples=100):
        print('Computing generative model specificity')
        min_distances = []
        for _ in tqdm.tqdm(range(n_samples)):
            sample = self.random_generation(1)

            mean_distances = []
            for data in self._train_loader:
                if self._config['data']['swap_features']:
                    x = data.x[self._manager.batch_diagonal_idx, ::]
                else:
                    x = data.x

                if self._normalized_data:
                    x = self._unnormalize_verts(x.to(self._device))
                else:
                    x = x.to(self._device)

                v_dist = self._manager.compute_vertex_errors(
                    x, sample.expand(x.shape[0], -1, -1))
                mean_distances.append(torch.mean(v_dist, dim=1))
            min_distances.append(torch.min(torch.cat(mean_distances, dim=0)))
        return torch.mean(torch.stack(min_distances)).item()

    def evaluate_gen(self, data_loader, n_sampled_points=None):
        all_sample = []
        all_ref = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            if self._normalized_data:
                data.x = self._unnormalize_verts(data.x)

            ref = data.x
            sample = self.random_generation(data.x.shape[0])

            if n_sampled_points is not None:
                subset_idxs = np.random.choice(ref.shape[1], n_sampled_points)
                ref = ref[:, subset_idxs]
                sample = sample[:, subset_idxs]

            all_ref.append(ref)
            all_sample.append(sample)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        print("Generation sample size:%s reference size: %s"
              % (sample_pcs.size(), ref_pcs.size()))

        # Compute metrics
        metrics = compute_all_metrics(
            sample_pcs, ref_pcs, self._config['optimization']['batch_size'])
        metrics = {k: (v.cpu().detach().item()
                       if not isinstance(v, float) else v) for k, v in
                   metrics.items()}
        print(metrics)

        sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        jsd = jsd_between_point_cloud_sets(sample_pcl_npy, ref_pcl_npy)
        print("JSD:%s" % jsd)
        metrics["jsd"] = jsd

        outfile_path = os.path.join(self._out_dir, 'eval_metrics_gen.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def latent_swapping(self, v_batch=None):
        if v_batch is None:
            v_batch = self.random_generation(2, denormalize=False)
        else:
            assert v_batch.shape[0] >= 2
            v_batch = v_batch.to(self._device)
            if self._config['data']['swap_features']:
                v_batch = v_batch[self._manager.batch_diagonal_idx, ::]
            v_batch = v_batch[:2, ::]

        z = self._manager.encode(v_batch)
        z_0, z_1 = z[0, ::], z[1, ::]

        swapped_verts = []
        for key, z_region in self._manager.latent_regions.items():
            z_swap = z_0.clone()
            z_swap[z_region[0]:z_region[1]] = z_1[z_region[0]:z_region[1]]
            swapped_verts.append(self._manager.generate(z_swap))

        all_verts = torch.cat([v_batch, torch.cat(swapped_verts, dim=0)], dim=0)

        if self._normalized_data:
            all_verts = self._unnormalize_verts(all_verts)

        out_mesh_dir = os.path.join(self._out_dir, 'latent_swapping')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        self.save_batch(all_verts, out_mesh_dir)

        source_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[0, ::].expand(all_verts.shape[0], -1, -1))
        target_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[1, ::].expand(all_verts.shape[0], -1, -1))

        renderings = self._manager.render(all_verts)
        renderings_source = self._manager.render(all_verts, source_dist, 5)
        renderings_target = self._manager.render(all_verts, target_dist, 5)
        grid = make_grid(torch.cat(
            [renderings, renderings_source, renderings_target], dim=-2),
            padding=10, pad_value=1, nrow=renderings.shape[0])
        save_image(grid, os.path.join(out_mesh_dir, 'latent_swapping.png'))

    def fit_vertices(self, target_verts, lr=5e-3, iterations=250,
                     target_noise=0, target_landmarks=None):
        # Scale and position target_verts
        target_verts = target_verts.unsqueeze(0).to(self._device)
        if target_landmarks is None:
            target_landmarks = target_verts[:, self.coma_landmarks, :]
        target_landmarks = target_landmarks.to(self._device)

        if target_noise > 0:
            target_verts = target_verts + (torch.randn_like(target_verts) *
                                           target_noise /
                                           self._manager.to_mm_const)
            target_landmarks = target_landmarks + (
                torch.randn_like(target_landmarks) *
                target_noise / self._manager.to_mm_const)

        z = self.latent_stats['means'].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            if i < iterations // 3:
                er = self._manager.compute_mse_loss(
                    gen_verts[:, self.uhm_landmarks, :], target_landmarks)
            else:
                er, _ = pytorch3d.loss.chamfer_distance(gen_verts, target_verts)

            er.backward()
            optimizer.step()
        return gen_verts, target_verts.squeeze()

    def fit_coma_data(self, base_dir='meshes2fit',
                      noise=0, export_meshes=False):
        print(f"Fitting CoMA meshes with noise = {noise} mm")
        out_mesh_dir = os.path.join(self._out_dir, 'fitting')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        names_and_scale = {}
        for dirpath, _, fnames in os.walk(base_dir):
            for f in fnames:
                if f.endswith('.ply'):
                    if f[:5] in ['03274', '03275', '00128', '03277']:
                        names_and_scale[f] = 9
                    else:
                        names_and_scale[f] = 8

        dataframes = []
        for m_id, scale in tqdm.tqdm(names_and_scale.items()):
            df_id = m_id.split('.')[0]
            subd = False
            mesh_path = os.path.join(base_dir, m_id)
            target_mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
            target_verts = torch.tensor(
                target_mesh.vertices, dtype=torch.float,
                requires_grad=False, device=self._device)

            # scale and translate to match template. Values manually computed
            target_verts *= scale
            target_verts[:, 1] += 0.15

            # If target mesh was subdivided use original target to retrieve its
            # landmarks
            target_landmarks = None
            if 'subd' in m_id:
                subd = True
                df_id = m_id.split('_')[0]
                base_path = os.path.join(base_dir, m_id.split('_')[0] + '.ply')
                base_mesh = trimesh.load_mesh(base_path, 'ply', process=False)
                base_verts = torch.tensor(
                    base_mesh.vertices, dtype=torch.float,
                    requires_grad=False, device=self._device)
                target_landmarks = base_verts[self.coma_landmarks, :]
                target_landmarks = target_landmarks.unsqueeze(0)
                target_landmarks *= scale
                target_landmarks[:, 1] += 0.15

            out_verts, t_verts = self.fit_vertices(
                target_verts, target_noise=noise,
                target_landmarks=target_landmarks)

            closest_p_errors = self._manager.to_mm_const * \
                self._dist_closest_point(out_verts, target_verts.unsqueeze(0))

            dataframes.append(pd.DataFrame(
                {'id': df_id, 'noise': noise, 'subdivided': subd,
                 'errors': closest_p_errors.squeeze().detach().cpu().numpy()}))

            if export_meshes:
                mesh_name = m_id.split('.')[0]
                out_mesh = trimesh.Trimesh(
                    out_verts[0, ::].cpu().detach().numpy(),
                    self._manager.template.face.t().cpu().numpy())
                out_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_fit_{str(noise)}" + '.ply'))
                target_mesh.vertices = t_verts.detach().cpu().numpy()
                target_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_t_{str(noise)}" + '.ply'))
        return pd.concat(dataframes)

    def fit_coma_data_different_noises(self, base_dir='meshes2fit'):
        noises = [0, 2, 4, 6, 8]
        dataframes = []
        for n in noises:
            dataframes.append(self.fit_coma_data(base_dir, n, True))
        df = pd.concat(dataframes)
        df.to_pickle(os.path.join(self._out_dir, 'coma_fitting.pkl'))

        sns.set_theme(style="ticks")
        plt.figure()
        sns.lineplot(data=df, x='noise', y='errors',
                     markers=True, dashes=False, ci='sd')
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting.svg'))

        plt.figure()
        sns.boxplot(data=df, x='noise', y='errors', showfliers=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_box.svg'))

        plt.figure()
        sns.violinplot(data=df[df.errors < 3], x='noise', y='errors',
                       split=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_violin.svg'))

    @staticmethod
    def _point_mesh_distance(points, verts, faces):
        points = points.squeeze()
        verts_packed = verts.to(points.device)
        faces_packed = torch.tensor(faces, device=points.device).t()
        first_idx = torch.tensor([0], device=points.device)

        tris = verts_packed[faces_packed]

        point_to_face = point_face_distance(points, first_idx, tris,
                                            first_idx, points.shape[0])
        return point_to_face / points.shape[0]

    @staticmethod
    def _dist_closest_point(x, y):
        # for each point on x return distance to closest point in y
        x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        cham_x = x_nn.dists[..., 0]
        return cham_x

    def direct_manipulation(self, z=None, indices=None, new_coords=None,
                            lr=0.1, iterations=50, affect_only_zf=True):
        if z is None:
            z = self.latent_stats['means'].unsqueeze(0)
            # z = self.random_latent(1)
            z = z.clone().detach().requires_grad_(True)
        if indices is None and new_coords is None:
            indices = [8816, 8069, 8808]
            new_coords = torch.tensor([[-0.0108174, 0.0814601, 0.664498],
                                       [-0.1821480, 0.0190682, 0.419531],
                                       [-0.0096422, 0.3058790, 0.465528]])
        new_coords = new_coords.unsqueeze(0).to(self._device)

        colors = self._manager.template.colors.to(torch.long)
        features = [str(colors[i].cpu().detach().numpy()) for i in indices]
        assert all(x == features[0] for x in features)

        zf_idxs = self._manager.latent_regions[features[0]]

        optimizer = torch.optim.Adam([z], lr)
        initial_verts = self._manager.generate_for_opt(z.to(self._device))
        if self._normalized_data:
            initial_verts = self._unnormalize_verts(initial_verts)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            loss = self._manager.compute_mse_loss(
                gen_verts[:, indices, :], new_coords)
            loss.backward()

            if affect_only_zf:
                z.grad[:, :zf_idxs[0]] = 0
                z.grad[:, zf_idxs[1]:] = 0
            optimizer.step()

        # Save output meshes
        out_mesh_dir = os.path.join(self._out_dir, 'direct_manipulation')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        initial_mesh = trimesh.Trimesh(
            initial_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        initial_mesh.export(os.path.join(out_mesh_dir, 'initial.ply'))

        new_mesh = trimesh.Trimesh(
            gen_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        new_mesh.export(os.path.join(out_mesh_dir, 'new.ply'))

        for i, coords in zip(indices, new_coords[0, ::].detach().cpu().numpy()):
            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices = sphere.vertices + coords
            sphere.export(os.path.join(out_mesh_dir, f'target_{i}.ply'))

            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices += initial_verts[0, i, :].cpu().detach().numpy()
            sphere.export(os.path.join(out_mesh_dir, f'selected_{i}.ply'))

    def interpolate(self):
        with open(os.path.join('precomputed', f'data_split_{self._data_type}.json'), 'r') as fp:
            data = json.load(fp)
        test_list = data['test']
        meshes_root = self._test_loader.dataset.root

        # Pick first test mesh and find most different mesh in test set
        v_1 = None
        distances = [0]
        for i, fname in enumerate(test_list):
            mesh_path = os.path.join(meshes_root, fname)
            mesh = trimesh.load_mesh(mesh_path, process=False)
            mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                      requires_grad=False, device='cpu')
            if i == 0:
                v_1 = mesh_verts
            else:
                distances.append(
                    self._manager.compute_mse_loss(v_1, mesh_verts).item())

        m_2_path = os.path.join(
            meshes_root, test_list[np.asarray(distances).argmax()] + '.ply')
        m_2 = trimesh.load_mesh(m_2_path, 'ply', process=False)
        v_2 = torch.tensor(m_2.vertices, dtype=torch.float, requires_grad=False)

        v_1 = (v_1 - self._norm_dict['mean']) / self._norm_dict['std']
        v_2 = (v_2 - self._norm_dict['mean']) / self._norm_dict['std']

        z_1 = self._manager.encode(v_1.unsqueeze(0).to(self._device))
        z_2 = self._manager.encode(v_2.unsqueeze(0).to(self._device))

        features = list(self._manager.template.feat_and_cont.keys())

        # Interpolate per feature
        if self._config['data']['swap_features']:
            z = z_1.repeat(len(features) // 2, 1)
            all_frames, rows = [], []
            for feature in features:
                zf_idxs = self._manager.latent_regions[feature]
                z_1f = z_1[:, zf_idxs[0]:zf_idxs[1]]
                z_2f = z_2[:, zf_idxs[0]:zf_idxs[1]]
                z[:, zf_idxs[0]:zf_idxs[1]] = self.vector_linspace(
                    z_1f, z_2f, len(features) // 2).to(self._device)

                gen_verts = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)

                renderings = self._manager.render(gen_verts).cpu()
                all_frames.append(renderings)
                rows.append(make_grid(renderings, padding=10,
                            pad_value=1, nrow=len(features)))
                z = z[-1, :].repeat(len(features) // 2, 1)

            save_image(
                torch.cat(rows, dim=-2),
                os.path.join(self._out_dir, 'interpolate_per_feature.png'))
            write_video(
                os.path.join(self._out_dir, 'interpolate_per_feature.mp4'),
                torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate per variable
        z = z_1.repeat(3, 1)
        all_frames = []
        for z_i in range(self._manager.model_latent_size):
            z_1f = z_1[:, z_i]
            z_2f = z_2[:, z_i]
            z[:, z_i] = torch.linspace(z_1f.item(),
                                       z_2f.item(), 3).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            renderings = self._manager.render(gen_verts).cpu()
            all_frames.append(renderings)
            z = z[-1, :].repeat(3, 1)

        write_video(
            os.path.join(self._out_dir, 'interpolate_per_variable.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate all features
        zs = self.vector_linspace(z_1, z_2, len(features))

        gen_verts = self._manager.generate(zs.to(self._device))
        if self._normalized_data:
            gen_verts = self._unnormalize_verts(gen_verts)

        renderings = self._manager.render(gen_verts).cpu()
        im = make_grid(renderings, padding=10, pad_value=1, nrow=len(features))
        save_image(im, os.path.join(self._out_dir, 'interpolate_all.png'))


    # AGE TESTS

    def get_z(self, batch):

        """
        
        This function computes z

        """

        z = self._manager.encode(batch.to(self._device)).detach()

        # mu, age = self._manager.encode(batch.to(self._device))
        
        # mu = mu.detach()
        # age = age.detach()
 
        # z = torch.cat((mu, age), dim=1)

        return z
    
    def age_latent_changing(self, data_loader):

        """
        
        This function generates an image of 4 subjects from a batch and changes the age latent to 5 differenet ages and displays them with their difference maps compared to the orginal mesh. 
        A subjects goes through the encoder, the age latent is changed, then goes through the decoder. 

        Output: image of a batch with 5 different ages & difference maps when compared to the:
         - original
         - previous
         - first column

        """

        storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
        age_latent_ranges = self._config['testing']['age_latent_changing']
        age_latent_ranges_original = age_latent_ranges.copy()
        for i in range(len(age_latent_ranges)):
            age_latent_ranges[i] = (age_latent_ranges[i] - age_train_mean) / age_train_std

        batch = next(iter(data_loader))

        mesh_ids = batch.fname
        original_ages = batch.age.numpy()

        if self._config['data']['swap_features']:
            batch = batch.x[self._manager.batch_diagonal_idx, ::] 
        else:
            batch = batch.x

        z = self.get_z(batch)

        z_all = []
        z_all_prev = []
        z_all_first = []
        z_original = []

        error_scale = 10

        grid_nrows = len(age_latent_ranges)

        encoded_original_ages = []

        for i in tqdm.tqdm(range(z.shape[0])):
            # append pre model mesh
            original = batch[i, ::].clone().unsqueeze(0)
            if self._normalized_data:
                    original = self._unnormalize_verts(original.to(self._device))
            rendering_original = self._manager.render(original).cpu()
            z_original.append(rendering_original.squeeze())

            new_z = z[i, ::].clone()
            new_z_original = z[i, ::].clone()
            gen_verts_original = self._manager.generate(new_z_original.to(self._device))
            differences = []
            differences_prev = []
            differences_first = []
            differences_original = []
            age_latent_ranges[0] = new_z[-1]
            encoded_original_age = age_latent_ranges[0].item()
            encoded_original_age = round((encoded_original_age * age_train_std) + age_train_mean)
            encoded_original_ages.append(encoded_original_age)

            for j in range(grid_nrows):
                new_z[-1] = age_latent_ranges[j]
                gen_verts = self._manager.generate(new_z.to(self._device))

                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)
                    gen_verts_original = self._unnormalize_verts(gen_verts_original)


                renderings = self._manager.render(gen_verts).cpu()
                z_all.append(renderings.squeeze())
                if j == 0:
                    z_original.append(renderings.squeeze())
                else:
                    z_all_prev.append(renderings.squeeze())
                    z_all_first.append(renderings.squeeze())

                # difference from original
                differences_from_original = self._manager.compute_vertex_errors(gen_verts, gen_verts_original)
                differences_renderings = self._manager.render(gen_verts, differences_from_original, error_max_scale=error_scale).cpu().detach()
                differences.append(differences_renderings.squeeze())

                # difference from previous
                if j == 0:
                    pass
                elif j == 1:
                    differences_from_previous = self._manager.compute_vertex_errors(gen_verts, gen_verts)
                    differences_renderings_prev = self._manager.render(gen_verts, differences_from_previous, error_max_scale=error_scale).cpu().detach()
                    differences_prev.append(differences_renderings_prev.squeeze())
                else:
                    differences_from_previous = self._manager.compute_vertex_errors(gen_verts, gen_verts_prev)
                    differences_renderings_prev = self._manager.render(gen_verts, differences_from_previous, error_max_scale=error_scale).cpu().detach()
                    differences_prev.append(differences_renderings_prev.squeeze())

                # difference from first 
                if j == 0:
                    pass
                elif j == 1:
                    gen_verts_first = gen_verts
                    differences_from_first = self._manager.compute_vertex_errors(gen_verts, gen_verts)
                    differences_renderings_first = self._manager.render(gen_verts, differences_from_first, error_max_scale=error_scale).cpu().detach()
                    differences_first.append(differences_renderings_first.squeeze())
                else:
                    differences_from_first = self._manager.compute_vertex_errors(gen_verts, gen_verts_first)
                    differences_renderings_first = self._manager.render(gen_verts, differences_from_first, error_max_scale=error_scale).cpu().detach()
                    differences_first.append(differences_renderings_first.squeeze())

                # difference from pre-post encode/decode
                if j ==0:
                    pre_mesh = original
                    post_mesh = gen_verts_original
                    differences_from_pre_pre = self._manager.compute_vertex_errors(pre_mesh, pre_mesh)
                    differences_renderings_pre_pre = self._manager.render(pre_mesh, differences_from_pre_pre, error_max_scale=error_scale).cpu().detach()
                    differences_original.append(differences_renderings_pre_pre.squeeze())
                    differences_from_pre_post = self._manager.compute_vertex_errors(post_mesh, pre_mesh)
                    differences_renderings_pre_post = self._manager.render(gen_verts, differences_from_pre_post, error_max_scale=error_scale).cpu().detach()
                    differences_original.append(differences_renderings_pre_post.squeeze())

                
                gen_verts_prev = gen_verts

            z_all.extend(differences)
            z_all_prev.extend(differences_prev)
            z_all_first.extend(differences_first)
            z_original.extend(differences_original)

        # create a results file 

        line_to_add_1 = 'age_latent_changing original ages: ' + str(original_ages)
        line_to_add_2 = 'age_latent_changing original ages after encoded: ' + str(encoded_original_ages)
        line_to_add_3 = 'age_latent_changing new ages: ' + str(age_latent_ranges_original)
        line_to_add_4 = 'age_latent_changing mesh ids: ' + str(mesh_ids)

        filename = os.path.join(self._out_dir, 'results.txt')

        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('')
        else:
            print(f"{filename} already exists.")

        with open(filename, 'a') as file:
            file.write(line_to_add_1)
            file.write('\n' * 2)
            file.write(line_to_add_2)
            file.write('\n' * 2)
            file.write(line_to_add_3)
            file.write('\n' * 2)
            file.write(line_to_add_4)
            file.write('\n' * 2)

        padding_value = 10

        # difference from original 
        stacked_frames = torch.stack(z_all)
        grid = make_grid(stacked_frames, padding=padding_value, pad_value=1, nrow=grid_nrows) 
        save_image(grid, os.path.join(self._out_dir, f'age_latent_changing_original_{age_latent_ranges_original}.png'))

        # difference from previous 
        stacked_frames_prev = torch.stack(z_all_prev)
        grid_prev = make_grid(stacked_frames_prev, padding=padding_value, pad_value=1, nrow=grid_nrows-1) 
        save_image(grid_prev, os.path.join(self._out_dir, f'age_latent_changing_previous_{age_latent_ranges_original}.png'))

        # difference from first 
        stacked_frames_first = torch.stack(z_all_first)
        grid_first = make_grid(stacked_frames_first, padding=padding_value, pad_value=1, nrow=grid_nrows-1) 
        save_image(grid_first, os.path.join(self._out_dir, f'age_latent_changing_first_{age_latent_ranges_original}.png'))

        # difference from original 
        stacked_frames_original = torch.stack(z_original)
        grid_original = make_grid(stacked_frames_original, padding=padding_value, pad_value=1, nrow=2) 
        save_image(grid_original, os.path.join(self._out_dir, f'pre_post_model.png'))

        # ##################

        # # generate a colour bar for /plots 

        # stacked_frames = torch.stack(z_all)
        # grid = make_grid(stacked_frames, padding=10, pad_value=1, nrow=grid_nrows) 

        # # Convert the tensor to a numpy array and transpose the dimensions for matplotlib
        # grid_np = grid.numpy().transpose((1, 2, 0))

        # plt.figure(figsize=(10, 10))
        # plt.imshow(grid_np, interpolation='nearest', vmin=0, vmax=error_scale, cmap='plasma')  # Set the colorbar range to 0-5 and colormap to 'plasma'
        # cbar = plt.colorbar(cmap='plasma')  # Set the colorbar colormap to 'plasma'
        # cbar.set_label('Error (mm)')
        # plt.savefig(os.path.join(self._out_dir, f'colour bar_{age_latent_ranges_original}.png'))
        # plt.close()

        # ##################

    def conditional_age_latent_changing(self, data_loader):

        """
        
        This function generates an image of 4 subjects from a batch and changes the age latent to 5 differenet ages and displays them with their difference maps compared to the orginal mesh. 
        A subjects goes through the encoder, the age latent is changed, then goes through the decoder. 

        Output: image of a batch with 5 different ages & difference maps when compared to the:
         - original
         - previous
         - first column

        """

        assert self._config['data']['swap_features'] == False

        storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
            
        age_latent_ranges = self._config['testing']['age_latent_changing']
        # age_latent_ranges_original = age_latent_ranges.copy()
        age_latent_ranges = (age_latent_ranges - age_train_mean) / age_train_std

        batch_original = next(iter(data_loader))

        gen_all = []
        error_scale = 10

        for i in range(len(age_latent_ranges)):

            batch = batch_original.clone()
            differences = []

            # adding age condition to input data
            if i == 0:
                data = self.conditional_data(batch)
            else:
                batch.norm_age = torch.full((batch.norm_age.size(0), 1), age_latent_ranges[i], dtype=torch.float64)
                data = self.conditional_data(batch)

            # encode and generate z
            z = self.get_z(data.x)

            # # add age condition to z latent representation
            # if i == 0:
            #     z[:, -1] = batch.norm_age.view(-1)
            # else:
            #     batch.norm_age = torch.full(batch.norm_age.size(), age_latent_ranges[i], dtype=torch.float64)
            #     z[:, -1] = batch.norm_age.view(-1)

            # decode and generate output mesh
            if i == 0:
                gen_verts_gt = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    gen_verts_gt = self._unnormalize_verts(gen_verts_gt)
                renderings = self._manager.render(gen_verts_gt).cpu()
                gen_all.append(renderings.squeeze())

                differences_from_gt = self._manager.compute_vertex_errors(gen_verts_gt, gen_verts_gt)
                differences_renderings = self._manager.render(gen_verts_gt, differences_from_gt, error_max_scale=error_scale).cpu().detach()
                differences.append(differences_renderings.squeeze())
                gen_all.extend(differences)
            
            else:
                gen_verts = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)
                renderings = self._manager.render(gen_verts).cpu()
                gen_all.append(renderings.squeeze())

                differences_from_gt = self._manager.compute_vertex_errors(gen_verts, gen_verts_gt)
                differences_renderings = self._manager.render(gen_verts, differences_from_gt, error_max_scale=error_scale).cpu().detach()
                differences.append(differences_renderings.squeeze())
                gen_all.extend(differences)
            

        # grid_nrows = len(age_latent_ranges)
        grid_nrows = 4

        # difference from original 
        stacked_frames = torch.stack(gen_all)
        stacked_frames = stacked_frames.view(-1, *stacked_frames.shape[2:])
        grid = make_grid(stacked_frames, padding=10, pad_value=1, nrow=grid_nrows)
        save_image(grid, os.path.join(self._out_dir, 'test_change_age.png'))
        # save_image(grid, os.path.join(self._out_dir, f'age_latent_changing_original_{age_latent_ranges_original}.png'))

    def age_encoder_check(self, train_loader, test_loader):

        """
        
        This function calculates the mean absolute difference between the actual age and the age latent after being passed through the encoder once. This checks how well the encoder generates the correct age. 

        Output: scatter plot of predicted age against ground truth age  
        
        """

        storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')

        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)

        for j in range(2):

            if j == 0:
                data_loader = train_loader
            else:
                data_loader = test_loader
        
            all_diff = []
            age_original = []
            age_predict = []


            for data in tqdm.tqdm(data_loader):

                if self._config['model']['conditional_vae']:
                    data = self.conditional_data(data)

                batch = data.x
                age = data.age.squeeze().tolist()
                # name = data.fname.squeeze().tolist()
                name = data.fname

                if self._config['data']['swap_features']:
                    batch = batch[self._manager.batch_diagonal_idx, ::]

                # z = self._manager.encode(batch.to(self._device)).detach().cpu()
                z = self.get_z(batch)

                z_age = z[:, -1]
                z_age = (z_age * age_train_std) + age_train_mean
                z_age = z_age.squeeze().tolist()

                age_diff = abs(torch.tensor(age) - torch.tensor(z_age))
                age_diff = age_diff.squeeze().tolist()

                dataset_remove = self._config['data']['dataset_remove']

                for i in range(z.shape[0]):
                    if name[i] in dataset_remove:
                        pass
                    else:
                        age_predict.append(z_age[i])
                        age_original.append(age[i])                    
                        all_diff.append(age_diff[i])
                
            average_diff = np.mean(all_diff)

            if j == 0:
                train_age_original = age_original
                train_age_predict = age_predict
                train_average_diff = average_diff
            else:
                test_age_original = age_original
                test_age_predict = age_predict
                test_average_diff = average_diff

        # plot graph of actual age against age latent from encoder

        age_range = self._config['data']['dataset_age_range']
        min_age, max_age = map(int, age_range.split('-'))

        plt.figure(figsize=(5, 5))

        plt.clf()

        plt.scatter(train_age_original, train_age_predict, color='yellow', marker='x', label='Train dataset')
        plt.scatter(test_age_original, test_age_predict, color='blue', label='Test dataset')
        # plt.plot([0, 100], [0, 100], 'r--')
        plt.plot([min_age, max_age], [min_age, max_age], 'r--')

        plt.title(f'Age prediction on age latent')
        plt.xlabel('Ground truth age (years)')
        # plt.title(f'Encoder accuracy of age latent ({age_range})')
        # plt.xlabel('Original age')
        plt.ylabel('Predicted age (years)')

        plt.text(0.25, 0.1, f'Mean absolute difference (train) = {round(train_average_diff,2)} years', transform=plt.gca().transAxes)
        plt.text(0.25, 0.05, f'Mean absolute difference (test) = {round(test_average_diff,2)} years', transform=plt.gca().transAxes)

        plt.legend(loc='upper left')

        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        plt.savefig(os.path.join(self._out_dir, f'age_encoder_check_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'age_encoder_check_{age_range}.svg'))

        return average_diff

    def age_prediction_encode_output(self, train_loader, test_loader):

        """
        
        This function encodes the subjects, changes the age latent to a random age, decodes, then encodes again and meaures if the second encoding can generate 
        the same age that it was assigned to after the first encoding. If the encoder check test shows the encoder works well, this tests will show how well the decoder performs.

        Output: two scatter plots. One for assigned age vs predicted age after second encoding and one for GT age vs predicted age after second encoding. Both the training and val/test set are plotted
        
        """

        for j in range(2):

            if j == 0:
                data_loader = train_loader
            else:
                data_loader = test_loader

            age_actuals = []
            age_latents = []
            age_preds = []
            names = []

            for batch in tqdm.tqdm(data_loader):

                if self._config['data']['swap_features']:
                    x = batch.x[self._manager.batch_diagonal_idx, ::]   
                else:
                    x = batch.x

                age = batch.age 
                name = batch.fname

                z = self.get_z(x)
                
                storage_path = os.path.join(self._manager._precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
                with open(storage_path, 'rb') as file:
                    age_train_mean, age_train_std = \
                        pickle.load(file)
                    
                which_data_remove = self._config['data']['dataset_remove']
                age_range = self._config['data']['dataset_age_range']
                # age_lower = int(age_range.split("-")[0])
                # age_upper = int(age_range.split("-")[1])
                age_lower, age_upper = map(int, age_range.split('-'))

                for i in tqdm.tqdm(range(z.shape[0])):
                    if name[i] in which_data_remove:
                        pass
                    else:
                        age_latent = random.randrange(age_lower, age_upper)
                        age_latents.append(age_latent)
                        age_latent_norm = (age_latent - age_train_mean) / age_train_std
                        z[i, -1] = age_latent_norm

                gen_verts = self._manager.generate(z.to(self._device))
                # z_2 = self._manager.encode(gen_verts.to(self._device)).detach().cpu()
                z_2 = self.get_z(gen_verts)

                for i in tqdm.tqdm(range(z.shape[0])):
                    if name[i] in which_data_remove:
                        pass
                    else:
                        age_pred = z_2[i][-1].item()
                        age_pred = (age_pred * age_train_std) + age_train_mean
                        age_preds.append(int(age_pred))
                        age_actual = age[i].item()
                        age_actuals.append(age_actual)
                        names.append(name[i])
                    

            average_pred_diff_latent_pred = np.mean(np.abs(np.array(age_latents) - np.array(age_preds)))
            average_pred_diff_actual_pred = np.mean(np.abs(np.array(age_actuals) - np.array(age_preds)))

            if j == 0:
                train_age_latents = age_latents
                train_age_preds = age_preds
                train_age_actuals = age_actuals
                train_average_pred_diff_latent_pred = average_pred_diff_latent_pred
                train_average_pred_diff_actual_pred = average_pred_diff_actual_pred
            else:
                test_age_latents = age_latents
                test_age_preds = age_preds
                test_age_actuals = age_actuals
                test_average_pred_diff_latent_pred = average_pred_diff_latent_pred
                test_average_pred_diff_actual_pred = average_pred_diff_actual_pred

        plt.figure(figsize=(7, 7))

        plt.clf()

        plt.scatter(train_age_latents, train_age_preds, color='yellow', marker='x', label='Train dataset')
        plt.scatter(test_age_latents, test_age_preds, color='orange', label='Tes dataset')
        plt.plot([age_lower, age_upper], [age_lower, age_upper], 'r--')

        plt.title(f'Age prediction on age latent using randomly assinged age')
        # plt.title(f'Decoder accuracy against random age ({age_range})')
        plt.xlabel('Random assigned age (years)')
        plt.ylabel('Predicted age (years)')

        plt.text(0.35, 0.1, f'Mean absolute difference (train) = {round(train_average_pred_diff_latent_pred, 2)} years', transform=plt.gca().transAxes)
        plt.text(0.35, 0.05, f'Mean absolute difference (test) = {round(test_average_pred_diff_latent_pred, 2)} years', transform=plt.gca().transAxes)

        plt.legend(loc='upper left')

        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_random_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_random_{age_range}.svg'))

        plt.figure(figsize=(7, 7))

        plt.clf()

        plt.scatter(train_age_actuals, train_age_preds, color='yellow', marker='x', label='Train dataset')
        plt.scatter(test_age_actuals, test_age_preds, color='orange', label='Test dataset')
        plt.plot([age_lower, age_upper], [age_lower, age_upper], 'r--')

        plt.title(f'Decoder accuracy against random age original age ({age_range})')
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')

        plt.text(0.35, 0.1, f'Mean absolute difference (train) = {round(train_average_pred_diff_actual_pred, 2)} years', transform=plt.gca().transAxes)
        plt.text(0.35, 0.05, f'Mean absolute difference (test) = {round(test_average_pred_diff_actual_pred, 2)} years', transform=plt.gca().transAxes)

        plt.legend(loc='upper left')

        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_original_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_original_{age_range}.svg'))

    def dataset_age_split(self, train_loader, val_loader, test_loader):

        """
        
        This function calculates how many data subjects there are for each age range group to understand it's distribution. 
        
        It saves the results in a .txt tile.  

        Output: two graphs: distribution of age & age split for train, val and test sets
        
        """

        # create lists with all ages and split

        precomputed_storage_path = self._config['data']['precomputed_path']

        data = [train_loader, val_loader, test_loader]
        data_name = ['train', 'val', 'test']
        data_type_list = []
        ages_list = []
        ages_list_original = []

        # storage_path = os.path.join(precomputed_storage_path, f'normalise_age_{self._data_type}.pkl')
        # with open(storage_path, 'rb') as file:
        #     age_train_mean, age_train_std = \
        #         pickle.load(file)

        for i in range(3):
            for batch in tqdm.tqdm(data[i]):  
                for j in range(batch.age.size()[0]):

                    # age = (batch.age[j].item() * age_train_std) + age_train_mean
                    age = batch.age[j].item()
                    ages_list_original.append(age)
                    if age == 0:
                        age = 0.001
                    else:
                        age = age-0.001
                    ages_list.append(age)
                    data_type_list.append(data_name[i])
        
        total_subjects = len(ages_list)

        # create age distribution graph of all data 

        age_range = self._config['data']['dataset_age_range']
        age_lower, age_upper = map(int, age_range.split('-'))

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_age_distribution_{age_range}.png')

        if self._data_type == 'lyhm':
            bins_num = 12
        else:
            bins_num = age_upper-age_lower + 1

        if not os.path.exists(storage_path):

            # bin_edges = list(range(0, 91, 10))
            # bin_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"]

            # bin_edges = np.linspace(age_lower, age_upper, bins_num)
            # bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

            # # Determine the overall min and max age across all datasets
            # overall_min_age = round(min(min(train_ages), min(test_ages), min(val_ages)))
            # overall_max_age = round(max(max(train_ages), max(test_ages), max(val_ages)))

            # Define the bin edges
            bin_edges = np.linspace(min(ages_list_original), max(ages_list_original), bins_num)
            bin_labels = [f"{int(bin_edges[i])}" for i in range(len(bin_edges))]
            label_points = [bin_edges[i] for i in range(len(bin_edges))]
            mid_points = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            

            # Plot the histogram - 0-2 mean up to and including 2 years old for example
            plt.figure(figsize=(10,6))
            n, bins, patches = plt.hist(ages_list, bins=bin_edges, alpha=0.7, edgecolor="k")
            # plt.hist(ages_list, bins=bin_edges, alpha=0.7)
            plt.xticks(label_points, bin_labels)
            plt.legend(loc='upper right')
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            plt.title("Age Distribution")

            # Annotate the bars with the frequency count and percentage
            for i in range(len(n)):
                plt.text(mid_points[i], n[i] + 5, f"{int(n[i])}\n({(n[i] / len(ages_list)) * 100:.1f}%)", 
                        ha='center', va='bottom', color='black', fontsize=9)

            plt.annotate(f'Total number of subjects: {total_subjects}', xy=(0.75, 0.95), xycoords='axes fraction')
            plt.tight_layout()
            plt.savefig(storage_path)
            plt.clf()

        else:
            print(f"{storage_path} already exists.")


        # create age split for train, val & test graph if it does not already exist 

        storage_path = os.path.join(precomputed_storage_path, f'{self._data_type}_data_split_{age_range}.png')

        if not os.path.exists(storage_path):

            # Initialize separate lists for train, test, and val ages
            train_ages = []
            test_ages = []
            val_ages = []
            train_ages_original = []
            test_ages_original  = []
            val_ages_original  = []

            # Iterate over data_types and ages simultaneously
            for data_type, age in zip(data_type_list, ages_list):
                if age == 0.001:
                    age_original = 0
                else: 
                    age_original = age + 0.001

                if data_type == 'train':
                    train_ages.append(age)
                    train_ages_original.append(age_original)
                elif data_type == 'test':
                    test_ages.append(age)
                    test_ages_original.append(age_original)
                elif data_type == 'val':
                    val_ages.append(age)
                    val_ages_original.append(age_original)

            plt.clf()


            # Determine the overall min and max age across all datasets
            overall_min_age = min(ages_list_original)
            overall_max_age = max(ages_list_original)

            # Define the bin edges
            bin_edges = np.linspace(overall_min_age, overall_max_age, bins_num)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.hist(train_ages, bins=bin_edges, alpha=0.5, label='Train', edgecolor='black')
            plt.hist(val_ages, bins=bin_edges, alpha=0.5, label='Validation', edgecolor='black')
            plt.hist(test_ages, bins=bin_edges, alpha=0.5, label='Test', edgecolor='black')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.title('Age distribution in Train, Validation and Test sets')

            # bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
            # mid_points = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            # plt.xticks(mid_points, bin_labels)

            bin_labels = [f"{int(bin_edges[i])}" for i in range(len(bin_edges))]
            label_points = [bin_edges[i] for i in range(len(bin_edges))]
            plt.xticks(label_points, bin_labels)

            # Annotate min and max for each set
            plt.annotate(f'Train min: {min(train_ages_original)}, max: {max(train_ages_original)}, count: {len(train_ages_original)}', xy=(0.55, 0.95), xycoords='axes fraction')
            plt.annotate(f'Validation min: {min(val_ages_original)}, max: {max(val_ages_original)}, count: {len(val_ages_original)}', xy=(0.55, 0.85), xycoords='axes fraction')
            plt.annotate(f'Test min: {min(test_ages_original)}, max: {max(test_ages_original)}, count: {len(test_ages_original)}', xy=(0.55, 0.90), xycoords='axes fraction')

            plt.savefig(storage_path)



    def process_data(self, loader):

        latents_list = []
        ages_list = []
        # fnames_list = []
        # data_list= []
        values_to_remove = self._config['data']['dataset_remove']

        for batch in tqdm.tqdm(loader):

            if self._config['data']['swap_features']:
                data = batch.x[self._manager.batch_diagonal_idx, ::] 
            else:
                data = batch.x
            
            z = self.get_z(data)
            z = z[:, :-1]

            ages = batch.age
            fnames = batch.fname

            for i in range(data.shape[0]):
                if fnames[i] in values_to_remove:
                    pass
                else:
                    ages_list.append(ages[i])
                    latents_list.append(z[i])

        ages = torch.cat(ages_list, dim=0)
        latents = torch.stack(latents_list).detach().cpu().numpy()

        return latents, ages

    

    def age_prediction_MLP(self, train_loader, val_loader):

        """ 
        
        This function uses an MLP to predict the age from the feature latents. 
        The feature z values are used for training that are generated by the VAE encoder. 
        The output is the average absolute difference between GT age and age predicted

        Output: graph plotting predicted vs actual age along with the mean absolute difference 
        
        """

        assert self._config['model']['age_disentanglement']

        # train_latents & val_latents are only the feature latents 
        # train_ages & val_ages are the ground truth ages
        train_latents, train_ages = self.process_data(train_loader)
        val_latents, val_ages = self.process_data(val_loader)
        # test_latents, test_ages = self.process_data(test_loader)

        # all_latents = np.concatenate((train_latents, val_latents), axis=0)
        # all_ages = np.concatenate((train_ages, val_ages), axis=0)

        sc = StandardScaler()
        train_latents_scaled = sc.fit_transform(train_latents)

        mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                            max_iter=300, activation='relu',
                            solver='adam', early_stopping=True,
                            validation_fraction=0.1, # % of data in validation set
                            n_iter_no_change=10, # Number of iterations with no improvement to wait before stopping
                            learning_rate_init=0.01)

        mlp_reg.fit(train_latents_scaled, train_ages)

        train_ages_pred = mlp_reg.predict(train_latents_scaled)
        train_age_diff = abs(train_ages - train_ages_pred)
        train_mean_age_diff = train_age_diff.mean()

        val_latents_scaled = sc.transform(val_latents)
        val_ages_pred = mlp_reg.predict(val_latents_scaled)
        val_age_diff = abs(val_ages - val_ages_pred)
        val_mean_age_diff = val_age_diff.mean()

        # create a graph plotting the actual ages against MLP predicted ages on test set
        age_range = self._config['data']['dataset_age_range']
        min_age, max_age = map(int, age_range.split('-'))

        plt.figure(figsize=(5, 5)) 

        plt.clf()

        plt.scatter(train_ages, train_ages_pred, color='yellow', marker='x', label='Train dataset')
        plt.scatter(val_ages, val_ages_pred, color='green', label='Test dataset')
        plt.plot([0, max_age], [0, max_age], 'r--')

        plt.title('Age prediction on feature latents')
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')

        plt.text(0.25, 0.1, f'Mean absolute difference (train) = {round(train_mean_age_diff.item(), 2)} years', transform=plt.gca().transAxes)
        plt.text(0.25, 0.05, f'Mean absolute difference (test) = {round(val_mean_age_diff.item(), 2)} years', transform=plt.gca().transAxes)

        plt.legend(loc='upper left')

        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        plt.savefig(os.path.join(self._out_dir, f'mlp_age_prediciton_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'mlp_age_prediciton_{age_range}.svg'))


    @staticmethod
    def vector_linspace(start, finish, steps):
        ls = []
        for s, f in zip(start[0], finish[0]):
            ls.append(torch.linspace(s, f, steps))
        res = torch.stack(ls)
        return res.t()


if __name__ == '__main__':
    import argparse
    import utils
    from data_generation_and_loading import get_data_loaders
    from model_manager import ModelManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='none',
                        help="ID of experiment")
    parser.add_argument('--output_path', type=str, default='.',
                        help="outputs path")
    opts = parser.parse_args()
    model_name = opts.id

    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')

    configurations = utils.get_config(
        os.path.join(output_directory, "config.yaml"))

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU not available, running on CPU")
    else:
        device = torch.device('cuda')
    
    if configurations['model']['age_disentanglement']:
        configurations['model']['latent_size'] += configurations['model']['age_latent_size']

    if configurations['model']['conditional_vae']:
        configurations['model']['in_channels'] += 1

    manager = ModelManager(
        configurations=configurations, device=device,
        precomputed_storage_path=configurations['data']['precomputed_path'])
    manager.resume(checkpoint_dir)

    train_loader, val_loader, test_loader, normalization_dict = \
        get_data_loaders(configurations, manager.template)

    tester = Tester(manager, normalization_dict, train_loader, val_loader, test_loader,
                    output_directory, configurations)

    tester()
    # tester.direct_manipulation()
    # tester.fit_coma_data_different_noises()
    # tester.set_renderings_size(512)
    # tester.set_rendering_background_color()
    # tester.interpolate()
    # tester.latent_swapping(next(iter(test_loader)).x)
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_specificity(train_loader, 100))
    # print(tester.compute_diversity_train_set())
    # print(tester.compute_diversity())
    # tester.evaluate_gen(test_loader, n_sampled_points=2048)

