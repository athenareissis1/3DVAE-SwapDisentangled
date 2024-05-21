import random
import torch

from torch_geometric.data import Data


class SwapFeatures:
    def __init__(self, template, model_config):
        self._template = template
        self._model_config = model_config
        self._features_and_contours = template.feat_and_cont
        self._zones_keys = list(template.feat_and_cont.keys())

    def __call__(self, batched_data):
        batch_size = batched_data.x.shape[0]
        if self._model_config['age_disentanglement']:
            batch_age = batched_data.age
            batch_norm_age = batched_data.norm_age
            batch_fname = batched_data.fname
        new_batch = torch.zeros([batch_size ** 2, *batched_data.x.shape[1:]],
                                device=batched_data.x.device,
                                dtype=batched_data.x.dtype)
        key = random.choice(self._zones_keys)
        for j in range(batch_size):
            for i in range(batch_size):
                if i == j:
                    new_batch[i * batch_size + j, ::] = batched_data.x[i, ::]
                else:
                    vertices = batched_data.x.numpy()
                    new_batch[i * batch_size + j, ::] = self.swap(
                        vertices[i, ::], vertices[j, ::], key)
        if self._model_config['age_disentanglement']:
            batched_data = Data(x=new_batch, swapped=key, age=batch_age, norm_age=batch_norm_age, fname=batch_fname)
        else: 
            batched_data = Data(x=new_batch, swapped=key)
        return batched_data

    def swap(self, verts0, verts1, feature_key):
        feature_idxs = self._features_and_contours[feature_key]['feature']

        verts0 = verts0.copy()
        verts1 = verts1.copy()

        feature_verts1 = verts1[feature_idxs]
        verts0[feature_idxs] = feature_verts1
        return torch.tensor(verts0)

