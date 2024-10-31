import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from bench3dgs.general_utils import strip_symmetric, build_scaling_rotation

import kornia
from plas import sort_with_plas
import torch.nn.functional as F
import timeit

def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inverse_log_transform(y):
    assert y.max() < 20, "Probably mixed up linear and log values for xyz. These going in here are supposed to be quite small (log scale)"
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.disable_xyz_log_activation:
            self.xyz_activation = lambda x: x
            self.inverse_xyz_activation = lambda x: x
        else:
            self.xyz_activation = inverse_log_transform
            self.inverse_xyz_activation = log_transform

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, disable_xyz_log_activation):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.disable_xyz_log_activation = disable_xyz_log_activation
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        activated = self.xyz_activation(self._xyz)
        return activated

    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.active_sh_degree == 0:
            return features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_attr_flat(self, attr_name):
        attr = getattr(self, f"_{attr_name}")
        return attr.flatten(start_dim=1)

    def get_activated_attr_flat(self, attr_name):
        getter_method = f"get_{attr_name}"
        attr = getattr(self, getter_method)
        return attr.flatten(start_dim=1)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        if self.active_sh_degree > 0:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
               l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



    # SSGS implementation

    def prune_all_but_these_indices(self, indices):

        if self.optimizer is not None:

            optimizable_tensors = self._prune_optimizer(indices)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[indices]
            self.denom = self.denom[indices]
            self.max_radii2D = self.max_radii2D[indices]
        else:
            self._xyz = self._xyz[indices]
            self._features_dc = self._features_dc[indices]
            self._features_rest = self._features_rest[indices]
            self._opacity = self._opacity[indices]
            self._scaling = self._scaling[indices]
            self._rotation = self._rotation[indices]


    def prune_to_square_shape(self, sort_by_opacity=True, verbose=True):
        num_gaussians = self._xyz.shape[0]

        self.grid_sidelen = int(np.sqrt(num_gaussians))
        num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen

        if verbose:
            print(f"Removing {num_removed}/{num_gaussians} gaussians to fit the grid. ({100 * num_removed / num_gaussians:.4f}%)")
        if self.grid_sidelen * self.grid_sidelen < num_gaussians:
            if sort_by_opacity:
                alpha = self.get_opacity[:, 0]
                _, keep_indices = torch.topk(alpha, k=self.grid_sidelen * self.grid_sidelen)
            else:
                shuffled_indices = torch.randperm(num_gaussians)
                keep_indices = shuffled_indices[:self.grid_sidelen * self.grid_sidelen]
            sorted_keep_indices = torch.sort(keep_indices)[0]
            self.prune_all_but_these_indices(sorted_keep_indices)

    @staticmethod
    def normalize(tensor):
        tensor = tensor - tensor.mean()
        if tensor.std() > 0:
            tensor = tensor / tensor.std()
        return tensor

    def sort_into_grid(self, sorting_cfg, verbose):
        
        normalization_fn = self.normalize if sorting_cfg.normalize else lambda x: x
        attr_getter_fn = self.get_activated_attr_flat if sorting_cfg.activated else self.get_attr_flat
        
        params_to_sort = []
        
        for attr_name, attr_weight in sorting_cfg.weights.items():
            if attr_weight > 0:
                params_to_sort.append(normalization_fn(attr_getter_fn(attr_name)) * attr_weight)
                    
        params_to_sort = torch.cat(params_to_sort, dim=1)
        
        if sorting_cfg.shuffle:
            shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)
            params_to_sort = params_to_sort[shuffled_indices]

        grid_to_sort = self.as_grid_img(params_to_sort).permute(2, 0, 1)

        start_time = timeit.default_timer()
        _, sorted_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
        duration = timeit.default_timer() - start_time
        
        sorted_indices = sorted_indices.squeeze().flatten()
        
        if sorting_cfg.shuffle:
            sorted_indices = shuffled_indices[sorted_indices]
        
        self.prune_all_but_these_indices(sorted_indices)

        return duration

    def as_grid_img(self, tensor):
        if not hasattr(self, "grid_sidelen"):
            raise "Gaussians not pruned yet!"

        if self.grid_sidelen * self.grid_sidelen != tensor.shape[0]:
            raise "Tensor shape does not match img sidelen, needs pruning?"

        img = tensor.reshape((self.grid_sidelen, self.grid_sidelen, -1))
        return img

    def attr_as_grid_img(self, attr_name):
        tensor = getattr(self, attr_name)
        return self.as_grid_img(tensor)

    def set_attr_from_grid_img(self, attr_name, img):

        if self.optimizer is not None:
            raise "Overwriting Gaussians during training not implemented yet! - Consider pruning method implementations"

        attr_shapes = {
            "_xyz": (3,),
            "_features_dc": (1, 3),
            "_features_rest": ((self.max_sh_degree + 1) ** 2 - 1, 3),
            "_rotation": (4,),
            "_scaling": (3,),
            "_opacity": (1,),
        }

        target_shape = attr_shapes[attr_name]
        img_shaped = img.reshape(-1, *target_shape)
        tensor = torch.tensor(img_shaped, dtype=torch.float, device="cuda")

        setattr(self, attr_name, tensor)

    def neighborloss_2d(self, tensor, neighbor_cfg, squeeze_dim=None):
        if neighbor_cfg.normalize:
            tensor = self.normalize(tensor)

        if squeeze_dim:
            tensor = tensor.squeeze(squeeze_dim)

        img = self.as_grid_img(tensor)
        img = img.permute(2, 0, 1).unsqueeze(0)

        blurred_x = kornia.filters.gaussian_blur2d(
            img.detach(),
            kernel_size=(1, neighbor_cfg.blur.kernel_size),
            sigma=(neighbor_cfg.blur.sigma, neighbor_cfg.blur.sigma),
            border_type="circular",
        )

        blurred_xy = kornia.filters.gaussian_blur2d(
            blurred_x,
            kernel_size=(neighbor_cfg.blur.kernel_size, 1),
            sigma=(neighbor_cfg.blur.sigma, neighbor_cfg.blur.sigma),
            border_type="reflect",
        )

        if neighbor_cfg.loss_fn == "mse":
            loss = F.mse_loss(blurred_xy, img)
        elif neighbor_cfg.loss_fn == "huber":
            loss = F.huber_loss(blurred_xy, img)
        else:
            assert False, "Unknown loss function"
        
        return loss

