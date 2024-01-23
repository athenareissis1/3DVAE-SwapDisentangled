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

from evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, val_load, test_load, out_dir, config):
        self._manager = model_manager
        self._manager.eval()
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._normalized_data = config['data']['normalize_data']
        self._extra_layers = config['data']['extra_layers']
        self._out_dir = out_dir
        self._config = config
        self._train_loader = train_load
        self._val_loader = val_load
        self._test_loader = test_load
        self._is_vae = self._manager.is_vae
        self.latent_stats = self.compute_latent_stats(train_load)
        

        self.coma_landmarks = [
            1337, 1344, 1163, 878, 3632, 2496, 2428, 2291, 2747,
            3564, 1611, 2715, 3541, 1576, 3503, 3400, 3568, 1519,
            203, 183, 870, 900, 867, 3536]
        self.uhm_landmarks = [
            10754, 10826, 9123, 10667, 19674, 28739, 4831, 19585,
            8003, 22260, 12492, 27386, 1969, 31925, 31158, 20963,
            1255, 9881, 32055, 45778, 5355, 27515, 18482, 33691]

    def __call__(self):
        # self.set_renderings_size(252)
        # self.set_rendering_background_color([1, 1, 1])

        # Qualitative evaluations
        # if self._config['data']['swap_features']:
        #     self.latent_swapping(next(iter(self._test_loader)).x)
        # self.per_variable_range_experiments(use_z_stats=False)
        # self.random_generation_and_rendering(n_samples=16)
        # self.random_generation_and_save(n_samples=16)
        # self.interpolate()
        # if self._config['data']['dataset_type'] == 'faces':
        #     self.direct_manipulation()

        # Quantitative evaluation
        # self.evaluate_gen(self._test_loader, n_sampled_points=2048)
        # recon_errors = self.reconstruction_errors(self._test_loader)
        # train_set_diversity = self.compute_diversity_train_set()
        # diversity = self.compute_diversity()
        # specificity = self.compute_specificity()
        # metrics = {'recon_errors': recon_errors,
        #            'train_set_diversity': train_set_diversity,
        #            'diversity': diversity,
        #            'specificity': specificity}

        # outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        # with open(outfile_path, 'w') as outfile:
        #     json.dump(metrics, outfile)


        # TEST TO RUN

        # # non-age tests
        # self.set_renderings_size(252)
        # self.set_rendering_background_color([1, 1, 1])
        # self.per_variable_range_experiments(use_z_stats=False)
        # self.random_generation_and_rendering(n_samples=16)

        # train_set_diversity = self.compute_diversity_train_set()
        diversity = self.compute_diversity()

        filename = os.path.join(self._out_dir, 'results.txt')
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('')

        with open(filename, 'a') as file:
            file.write(f"Train diversity: {str(train_set_diversity)}")
            file.write('\n' * 2)
            file.write(f"Diversity: {str(diversity)}")
            file.write('\n' * 2)
            

        # # age tests
        # self.age_latent_changing(self._test_loader, self._extra_layers)
        # self.age_encoder_check(self._test_loader)
        # self.age_prediction_MLP(self._train_loader, self._test_loader, self._extra_layers)
        # self.age_prediction_encode_output(self._test_loader, self._extra_layers)
        # self.dataset_age_split(self._train_loader, self._val_loader, self._test_loader)


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
        if self._is_vae and not use_z_stats:
            latent_size = self._manager.model_latent_size
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
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
        with open(os.path.join('precomputed', 'data_split.json'), 'r') as fp:
            data = json.load(fp)
        test_list = data['test']
        meshes_root = self._test_loader.dataset.root

        # Pick first test mesh and find most different mesh in test set
        v_1 = None
        distances = [0]
        for i, fname in enumerate(test_list):
            mesh_path = os.path.join(meshes_root, fname + '.ply')
            mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
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

    @staticmethod
    def vector_linspace(start, finish, steps):
        ls = []
        for s, f in zip(start[0], finish[0]):
            ls.append(torch.linspace(s, f, steps))
        res = torch.stack(ls)
        return res.t()

    # age related tests added here

    def get_z(self, batch, extra_mlp_layers):

        """
        
        This function computes z depending on if the extra linear layers have been added to model or not 

        """
        
        if extra_mlp_layers == False:
            z = self._manager.encode(batch.to(self._device)).detach()
        else:
            mu = self._manager.encode(batch.to(self._device)).detach()
            features = mu[:, :-1]
            age = mu[:, -1].unsqueeze(1)
            features_no_age = self._manager._net.mlp_feature_layers(features)
            z = torch.cat((features_no_age, age), dim=1)

        return z
        
    def age_latent_changing(self, data_loader, extra_layers):

        """
        
        This function generates an image of 4 subjects from a batch and changes the age latent to 5 differenet ages and displays them with their difference maps compared to the orginal mesh. 
        A subjects goes through the encoder, the age latent is changed, then goes through the decoder. 

        Output: image of a batch with 5 different ages & difference maps when compared to the:
         - original
         - previous
         - first column

        """

        batch = next(iter(data_loader))

        if self._config['data']['swap_features']:
                batch = batch.x[self._manager.batch_diagonal_idx, ::]        

        z = self.get_z(batch, extra_layers)

        z_all = []
        z_all_prev = []
        z_all_first = []

        error_scale = 10

        storage_path = os.path.join(self._manager._precomputed_storage_path, 'normalise_age.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
        age_latent_ranges = self._config['testing']['age_latent_changing']
        age_latent_ranges_original = age_latent_ranges.copy()
        for i in range(len(age_latent_ranges)):
            age_latent_ranges[i] = (age_latent_ranges[i] - age_train_mean) / age_train_std
        grid_nrows = len(age_latent_ranges)

        original_ages = []

        for i in tqdm.tqdm(range(z.shape[0])):
            new_z = z[i, ::].clone()
            new_z_original = z[i, ::].clone()
            gen_verts_original = self._manager.generate(new_z_original.to(self._device))
            differences = []
            differences_prev = []
            differences_first = []
            age_latent_ranges[0] = new_z[-1]
            original_age = age_latent_ranges[0].item()
            original_age = round((original_age * age_train_std) + age_train_mean)
            original_ages.append(original_age)

            for j in range(grid_nrows):
                new_z[-1] = age_latent_ranges[j]
                gen_verts = self._manager.generate(new_z.to(self._device))

                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)
                    gen_verts_original = self._unnormalize_verts(gen_verts_original)

                renderings = self._manager.render(gen_verts).cpu()
                z_all.append(renderings.squeeze())
                if j != 0:
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
                
                gen_verts_prev = gen_verts

            z_all.extend(differences)
            z_all_prev.extend(differences_prev)
            z_all_first.extend(differences_first)

        # create a results file 

        line_to_add_1 = 'age_latent_changing original ages: ' + str(original_ages)
        line_to_add_2 = 'age_latent_changing new ages: ' + str(age_latent_ranges_original)

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

        # difference from original 
        stacked_frames = torch.stack(z_all)
        grid = make_grid(stacked_frames, padding=10, pad_value=1, nrow=grid_nrows) 
        save_image(grid, os.path.join(self._out_dir, f'age_latent_changing_original_{age_latent_ranges_original}.png'))

        # difference from previous 
        stacked_frames_prev = torch.stack(z_all_prev)
        grid_prev = make_grid(stacked_frames_prev, padding=10, pad_value=1, nrow=grid_nrows-1) 
        save_image(grid_prev, os.path.join(self._out_dir, f'age_latent_changing_previous_{age_latent_ranges_original}.png'))

        # difference from first 
        stacked_frames_first = torch.stack(z_all_first)
        grid_first = make_grid(stacked_frames_first, padding=10, pad_value=1, nrow=grid_nrows-1) 
        save_image(grid_first, os.path.join(self._out_dir, f'age_latent_changing_first_{age_latent_ranges_original}.png'))

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
    
    def age_encoder_check(self, data_loader):

        """
        
        This function calculates the mean absolute difference between the actual age and the age latent after being passed through the encoder once. 

        Output: singe value in results.txt 
        
        """

        storage_path = os.path.join(self._manager._precomputed_storage_path, 'normalise_age.pkl')

        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
        
        all_diff = []
        age_original = []
        age_predict = []
        all_names = []
        for data in tqdm.tqdm(data_loader):

            batch = data.x
            if self._config['data']['swap_features']:
                batch = batch[self._manager.batch_diagonal_idx, ::]

            z = self._manager.encode(batch.to(self._device)).detach().cpu()

            name = data.name.squeeze().tolist()
            

            data_remove = self._config['data']['dataset_remove_outlier']
            which_data_remove = self._config['data']['dataset_remove']

            for i in range(z.shape[0]):
                z_age = z[:, -1]
                z_age = (z_age * age_train_std) + age_train_mean
                z_age = z_age.squeeze().tolist()

                age = (data.age * age_train_std) + age_train_mean
                age = age.squeeze().tolist()

                age_diff = abs(torch.tensor(age) - torch.tensor(z_age))
                age_diff = age_diff.squeeze().tolist()

                if name[i] in which_data_remove and data_remove:
                    pass
                else:
                    age_predict.append(z_age[i])
                    age_original.append(age[i])                    
                    all_diff.append(age_diff[i])
                    all_names.append(name[i])
            
        average_diff = np.mean(all_diff)

        # plot graph of actual age against age latent from encoder

        age_range = self._config['data']['dataset_age_range']

        plt.clf()

        plt.scatter(age_original, age_predict, color='blue')
        plt.plot([0, 100], [0, 100], 'r--')

        plt.title(f'Encoder accuracy of age latent ({age_range})')
        plt.xlabel('Original age')
        plt.ylabel('Predicted age')

        plt.text(0.1, 0.9, f'Mean absolute difference = {round(average_diff,2)}', transform=plt.gca().transAxes)

        plt.savefig(os.path.join(self._out_dir, f'age_encoder_check_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'age_encoder_check_{age_range}.svg'))

        return average_diff

    def age_prediction_MLP(self, train_loader, val_loader, extra_layers):

        """ 
        
        This function uses an MLP to predict the age from the latent vector after being passed through the endocer once. 
        The output is the average absolute difference between GT age and age predicted

        Output: single value in results.txt & graph plotting predicted vs actuall 
        
        """

        latents_list = []
        age_list = []
        for data in tqdm.tqdm(train_loader):
            age_list.append(data.age)
            if self._config['data']['swap_features']:
                data = data.x[self._manager.batch_diagonal_idx, ::] 

            z = self.get_z(data, extra_layers)

            if self._config['data']['age_disentanglement']:
                z = z[:, :-1]
            latents_list.append(z)
        train_latents = torch.cat(latents_list, dim=0)
        train_ages = torch.cat(age_list, dim=0)

        latents_list = []
        age_list = []
        file_names = []
        for data in tqdm.tqdm(val_loader):
            age_list.append(data.age)
            file_names.append(data.name)
            if self._config['data']['swap_features']:
                data = data.x[self._manager.batch_diagonal_idx, ::]

            z = self.get_z(data, extra_layers)

            if self._config['data']['age_disentanglement']:
                z = z[:, :-1]
            latents_list.append(z)
        val_latents = torch.cat(latents_list, dim=0)
        val_ages = torch.cat(age_list, dim=0)
        val_file_names = torch.cat(file_names, dim=0)

        train_latents_np = train_latents.detach().cpu().numpy()
        val_latents_np = val_latents.detach().cpu().numpy()

        sc = StandardScaler()
        scaler = sc.fit(train_latents_np)

        train_latents_scaled = scaler.transform(train_latents_np)
        val_latents_scaled = scaler.transform(val_latents_np)

        mlp_reg = MLPRegressor(hidden_layer_sizes = (150,100,50),
                        max_iter = 300, activation = 'relu',
                        solver = 'adam', early_stopping = True, learning_rate_init=0.01)
        

        mlp_reg.fit(train_latents_scaled, train_ages)
        val_ages_pred = mlp_reg.predict(val_latents_scaled)

        val_ages = val_ages.detach().cpu().numpy().flatten()
        val_ages_pred = val_ages_pred.flatten()
        val_file_names = val_file_names.detach().cpu().numpy().flatten()

        values_to_remove = self._config['data']['dataset_remove']

        for value in values_to_remove:
            index = np.where(val_file_names == value)[0]

            val_ages = np.delete(val_ages, index)
            val_ages_pred = np.delete(val_ages_pred, index)
            val_file_names = np.delete(val_file_names, index)

        storage_path = os.path.join(self._manager._precomputed_storage_path, 'normalise_age.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)
        
        age_diff = abs(val_ages - val_ages_pred) * age_train_std
        average_age_diff = np.mean(age_diff)

        # un-normalise age values

        val_ages = (val_ages * age_train_std) + age_train_mean
        val_ages_pred = (val_ages_pred * age_train_std) + age_train_mean

        # create a graph plotting the actual ages against MLP predicted ages on val set

        age_range = self._config['data']['dataset_age_range']

        plt.clf()

        plt.scatter(val_ages, val_ages_pred, color='green')
        plt.plot([0, 100], [0, 100], 'r--')

        plt.title('Age prediction on feature latents')
        plt.xlabel('Ground truth age')
        plt.ylabel('MLP predicted age')

        plt.text(0.1, 0.9, f'Mean absolute difference = {round(average_age_diff, 2)}', transform=plt.gca().transAxes)

        plt.savefig(os.path.join(self._out_dir, f'mlp_prediciton_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'mlp_prediciton_{age_range}.svg'))


        return average_age_diff

    def age_prediction_encode_output(self, data_loader, extra_layers):

        """
        
        This function encodes the subjects, changes the age latent, decodes, then encodes again and meaures if the second encoding can generate 
        the same age that it was assigned to after the first encoding.

        The output is the mean absolute difference after second encoder between the given age and the predicted age

        It also outputs a graph with the actual and predicted ages 

        Output: two lines in results.txt & two graphs
        
        """

        age_actuals = []
        age_latents = []
        age_preds = []
        names = []

        for batch in tqdm.tqdm(data_loader):

            if self._config['data']['swap_features']:
                    x = batch.x[self._manager.batch_diagonal_idx, ::]    
            age = batch.age 
            name = batch.name

            z = self.get_z(x, extra_layers)
            
            storage_path = os.path.join(self._manager._precomputed_storage_path, 'normalise_age.pkl')
            with open(storage_path, 'rb') as file:
                age_train_mean, age_train_std = \
                    pickle.load(file)
                
            data_remove = self._config['data']['dataset_remove_outlier']
            which_data_remove = self._config['data']['dataset_remove']
            age_range = self._config['data']['dataset_age_range']
            age_lower = int(age_range.split("-")[0])
            age_upper = int(age_range.split("-")[1])

            for i in tqdm.tqdm(range(z.shape[0])):
                if name[i].item() in which_data_remove and data_remove:
                    pass
                else:
                    age_latent = random.randrange(age_lower, age_upper)
                    age_latents.append(age_latent)
                    age_latent_norm = (age_latent - age_train_mean) / age_train_std
                    z[i, -1] = age_latent_norm

            gen_verts = self._manager.generate(z.to(self._device))
            z_2 = self._manager.encode(gen_verts.to(self._device)).detach().cpu()

            for i in tqdm.tqdm(range(z.shape[0])):
                if name[i].item() in which_data_remove and data_remove:
                    pass
                else:
                    age_pred = z_2[i][-1].item()
                    age_pred = (age_pred * age_train_std) + age_train_mean
                    age_preds.append(int(age_pred))
                    age_actual = age[i].item()
                    age_actual = (age_actual * age_train_std) + age_train_mean
                    age_actuals.append(age_actual)
                    names.append(name[i])
                

        average_pred_diff_latent_pred = np.mean(np.abs(np.array(age_latents) - np.array(age_preds)))
        average_pred_diff_actual_pred = np.mean(np.abs(np.array(age_actuals) - np.array(age_preds)))

        plt.clf()

        plt.scatter(age_latents, age_preds, color='orange')
        plt.plot([0, 100], [0, 100], 'r--')

        plt.title(f'Decoder accuracy against random age ({age_range})')
        plt.xlabel('Random assigned age')
        plt.ylabel('Predicted age')

        plt.text(0.1, 0.9, f'Mean absolute difference = {round(average_pred_diff_latent_pred, 2)}', transform=plt.gca().transAxes)

        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_random_{age_range}.png'))
        plt.savefig(os.path.join(self._out_dir, f'decoder_accuracy_random_{age_range}.svg'))

        plt.clf()

        plt.scatter(age_actuals, age_preds, color='orange')
        plt.plot([0, 100], [0, 100], 'r--')

        plt.title(f'Decoder accuracy against random age original age ({age_range})')
        plt.xlabel('Original age')
        plt.ylabel('Predicted age')

        plt.text(0.1, 0.9, f'Mean absolute difference = {round(average_pred_diff_actual_pred, 2)}', transform=plt.gca().transAxes)

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

        storage_path = os.path.join(precomputed_storage_path, 'normalise_age.pkl')
        with open(storage_path, 'rb') as file:
            age_train_mean, age_train_std = \
                pickle.load(file)

        for i in range(3):
            for batch in tqdm.tqdm(data[i]):  
                for j in range(batch.age.size()[0]):

                    age = (batch.age[j].item() * age_train_std) + age_train_mean
                    ages_list.append(age-0.01)
                    data_type_list.append(data_name[i])
        
        total_subjects = len(ages_list)

        # create age distribution graph of all data 

        age_range = self._config['data']['dataset_age_range']
        storage_path = os.path.join(precomputed_storage_path, f'age_distribution_{age_range}.png')

        if not os.path.exists(storage_path):

            bin_edges = list(range(0, 91, 10))
            bin_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"]
            mid_points = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

            # Plot the histogram
            plt.figure(figsize=(10,6))
            n, bins, patches = plt.hist(ages_list, bins=bin_edges, alpha=0.7, edgecolor="k")
            # plt.hist(ages_list, bins=bin_edges, alpha=0.7)
            plt.xticks(mid_points, bin_labels)
            plt.legend(loc='upper right')
            plt.xlabel("Age Ranges")
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

        storage_path = os.path.join(precomputed_storage_path, f'data_split_{age_range}.png')

        if not os.path.exists(storage_path):

            # Initialize separate lists for train, test, and val ages
            train_ages = []
            test_ages = []
            val_ages = []

            # Iterate over data_types and ages simultaneously
            for data_type, age in zip(data_type_list, ages_list):
                if data_type == 'train':
                    train_ages.append(age+0.01)
                elif data_type == 'test':
                    test_ages.append(age+0.01)
                elif data_type == 'val':
                    val_ages.append(age+0.01)

            plt.clf()

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.hist(train_ages, bins=20, alpha=0.5, label='Train')
            plt.hist(test_ages, bins=20, alpha=0.5, label='Test')
            plt.hist(val_ages, bins=20, alpha=0.5, label='Validation')
            plt.legend(loc='upper right')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.title('Age distribution in Train, Test and Validation sets')

            # Annotate min and max for each set
            plt.annotate(f'Train min: {min(train_ages)}, max: {max(train_ages)}', xy=(0.5, 0.95), xycoords='axes fraction')
            plt.annotate(f'Validation min: {min(val_ages)}, max: {max(val_ages)}', xy=(0.5, 0.85), xycoords='axes fraction')
            plt.annotate(f'Test min: {min(test_ages)}, max: {max(test_ages)}', xy=(0.5, 0.90), xycoords='axes fraction')

            plt.savefig(storage_path)

        else:
            print(f"{storage_path} already exists.")

    def proportions_check(self):
        
        return 0

    def change_other_latents(self, data_loader):

        """
        
        This function shows how changing the non-age latent variables has an effect on the output  

        Output: multiple images fo each change
        
        """

        batch = next(iter(data_loader))

        if self._config['data']['swap_features']:
                batch = batch.x[self._manager.batch_diagonal_idx, ::]

        enc = self._manager.encode(batch.to(self._device)).detach().cpu()

        rand = random.randint(0, len(batch) - 1)

        z = enc[rand, ::]     
        # import pdb; pdb.set_trace()

        for i in range(55):

            z_min = z.clone()
            z_max = z.clone()
            # z_latent_index = i + 1
            # z_latent_value = z[i] 

            z_min[i] = -3 
            z_max[i] = 3

            z_all = [z_min, z, z_max]

            gen_all = []
            differences = []

            for j in range(3):

                gen = self._manager.generate(z_all[j].to(self._device))

                if self._normalized_data:
                    gen = self._unnormalize_verts(gen)

                renderings = self._manager.render(gen).cpu()
                gen_all.append(renderings.squeeze())

                differences_from_original = self._manager.compute_vertex_errors(gen_verts, gen_verts_original)
                differences_renderings = self._manager.render(gen_verts, differences_from_original, error_max_scale=5).cpu().detach()
                differences.append(differences_renderings.squeeze())

            # difference from one to two 

            gen_all.extend(differences)

            stacked_frames = torch.stack(gen_all)
            grid = make_grid(stacked_frames, padding=10, pad_value=1, nrow=3) 
            save_image(grid, os.path.join(self._out_dir + '/latent_changes', str(i+1) + '_latent_changed.png'))


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
    
    if configurations['data']['age_disentanglement'] == True:
        configurations['model']['latent_size'] = configurations['model']['latent_size'] + 1

    extra_layers = configurations['data']['extra_layers']

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU not available, running on CPU")
    else:
        device = torch.device('cuda')

    manager = ModelManager(
        configurations=configurations, device=device,
        precomputed_storage_path=configurations['data']['precomputed_path'])
    manager.resume(checkpoint_dir)

    train_loader, val_loader, test_loader, normalization_dict = \
        get_data_loaders(configurations, manager.template)

    tester = Tester(manager, normalization_dict, train_loader, val_loader, test_loader,
                    output_directory, configurations)

    # # tester()
    # tester.direct_manipulation()
    # tester.fit_coma_data_different_noises()
    # tester.set_renderings_size(256)
    # tester.set_rendering_background_color()
    # tester.interpolate()
    # tester.latent_swapping(next(iter(test_loader)).x)
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_specificity(train_loader, 100))
    # print(tester.compute_diversity_train_set())
    print(tester.compute_diversity())
    # tester.evaluate_gen(test_loader, n_sampled_points=2048)

    # TESTS TO RUN

    # non-age tests
    # tester.set_renderings_size(256)
    # tester.set_rendering_background_color()
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)

    # # age tests
    # if configurations['data']['age_disentanglement'] == True:
        # tester.age_latent_changing(test_loader, extra_layers)
        # tester.age_encoder_check(test_loader)
        # tester.age_prediction_MLP(train_loader, test_loader, extra_layers)
        # tester.age_prediction_encode_output(test_loader, extra_layers)
        # tester.dataset_age_split(train_loader, val_loader, test_loader)
