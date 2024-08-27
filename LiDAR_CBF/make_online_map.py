from hocbf_composition.barrier import Barrier
from hocbf_composition.utils.utils import *
from LiDAR_CBF.utils.utils import *
from LiDAR_CBF.utils.smooth_function_maker import SmoothFunction
from LiDAR_CBF.utils.piecewise_function import DynamicPiecewiseFunction
from hocbf_composition.dynamics import AffineInControlDynamics




class LidarMap:
    def __init__(self, lidar, dynamics, cfg):
        self.lidar = lidar
        self.cfg = cfg
        self.smooth_function = SmoothFunction(cfg, lidar.lidar_cfg.update_rate)
        self._make_time_augmented_dynamics(dynamics)

        assert issubclass(globals()[cfg.cbf_synthesis], CBFSynthesis), "Synthesis method not implemented for this class"
        self.cbf_synthesis = globals()[cfg.cbf_synthesis](lidar, cfg)

        self.buffer = []
        self.barrier = None

    def initialize(self, pos, point_cloud):
        local_cbf = self.cbf_synthesis.update(pos, point_cloud)
        for _ in range(self.cfg.memory):
            self.buffer.append(local_cbf)

    def update(self, pos, point_cloud, robot_indices=None):
        if len(self.buffer) == 0:
            self.initialize(pos, point_cloud)
        local_cbf = self.cbf_synthesis.update(pos, point_cloud)
        self.buffer.append(local_cbf)
        self.barrier = self.make_pos_barrier(self.buffer[-self.cfg.memory-1:], robot_indices)

    def make_pos_barrier(self, buffer, robot_indices=None):
        return (Barrier().assign(barrier_func=self.make_pos_barrierfunc_from_local_cbf(buffer, robot_indices),
                                       rel_deg=self.cfg.pos_barrier_rel_deg,
                                       alphas=make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha))
                .assign_dynamics(self.dynamics))


    def _make_time_augmented_dynamics(self, dynamics):
        aug_state_dim = dynamics.state_dim + 1  # +1 for time
        aug_action_dim = dynamics.action_dim

        self.dynamics = AffineInControlDynamics(state_dim=aug_state_dim,
                                                 action_dim=aug_action_dim)

        def aug_f(x):
            return torch.hstack((
                torch.ones(x.shape[0], 1),  # Time derivative is always 1
                dynamics.f(x=x[:, 1:])
            ))

        self.dynamics.set_f(aug_f)

        def aug_g(x):
            return torch.cat((
                torch.zeros(x.shape[0], 1, self.dynamics.action_dim, dtype=torch.float64),
                dynamics.g(x[:, 1:])
            ), dim=1)

        self.dynamics.set_g(aug_g)

    def make_pos_barrierfunc_from_local_cbf(self, memory, robot_indices=None):
        if self.cfg.memory == 1:
            def barrier_func(x):
                x = vectorize_input(x)
                eta = self.smooth_function(x[..., 0:1])
                return (eta * memory[-1](x[..., 1:3], robot_indices) +
                        (1 - eta) * memory[0](x[..., 1:3], robot_indices)).permute(1, 0, 2).reshape(-1, 1)
        else:
            buffer_funcs = list(memory)[1:-1]

            def barrier_func(x):
                x = vectorize_input(x)
                eta = self.smooth_function(x[..., 0:1])
                tv_output = eta * memory[-1](x[..., 1:3], robot_indices) + (1 - eta) * memory[0](x[..., 1:3],
                                                                                                 robot_indices)

                func_outputs = [func(x[..., 1:3], robot_indices) for func in buffer_funcs] + [tv_output]
                stacked_outputs = torch.stack(func_outputs, dim=-1)

                return softmax(stacked_outputs, self.cfg.softmax_rho, dim=-1).permute(1, 0, 2).reshape(-1, 1)

        return barrier_func

    def make_piecewise_barrier(self, dynamics=None, robot_indices=None):
        dynamics = dynamics if dynamics is not None else self.dynamics

        buffer_len = len(self.buffer)
        memory_size = self.cfg.memory
        self.pw_barrier = []

        for i in robot_indices:
            pos_pw_barrierfunc = DynamicPiecewiseFunction(self.lidar.lidar_cfg.update_rate)

            # Pre-compute barrier functions for all slices
            barrier_funcs = [
                self.make_pos_barrierfunc_from_local_cbf(
                    self.buffer[j:j + memory_size + 1],
                    [i]  # Pass a single robot index
                )
                for j in range(buffer_len - memory_size + 1)
            ]

            # Add functions efficiently
            for func in barrier_funcs:
                pos_pw_barrierfunc.add_func(func)

            # Create the barrier object
            barrier = (
                Barrier()
                .assign(
                    barrier_func=pos_pw_barrierfunc,
                    rel_deg=self.cfg.pos_barrier_rel_deg,
                    alphas=make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
                )
                .assign_dynamics(dynamics)
            )

            self.pw_barrier.append(barrier)





class CBFSynthesis:
    def __init__(self, lidar, cfg):
        self.lidar = lidar
        self.cfg = cfg

    def update(self, pos, point_cloud):
        raise NotImplementedError


class EllipseCBF(CBFSynthesis):


    def update(self, pos, point_cloud):
        circle_func = self._make_circle(pos)
        if torch.isnan(point_cloud['detected_points']).all():
            return lambda x, robot_indices: circle_func(x, robot_indices).unsqueeze(-1)
        else:
            ellipses_func = self._make_ellipse(point_cloud['boundary_points'], point_cloud['detected_points'])

            def lvlset_generator(x, robot_indices):
                ellipses = ellipses_func(x, robot_indices)
                circle = circle_func(x, robot_indices).unsqueeze(-1)
                nan_indices = torch.repeat_interleave(torch.isnan(point_cloud['detected_points'])[robot_indices, ..., 0]
                                                      , ellipses.shape[0], dim=0)
                if nan_indices.ndim == 2:
                    nan_indices = nan_indices.unsqueeze(1)

                ellipses[nan_indices] = torch.repeat_interleave(circle, ellipses.shape[-1], dim=-1)[nan_indices]
                return (softmin(torch.cat([ellipses, circle], dim=-1), self.cfg.softmin_rho, dim=-1)
                        .unsqueeze(-1))

            return lvlset_generator

    def _make_ellipse(self, maxrange, boundary):
        boundary = torch.where(torch.isnan(boundary), maxrange, boundary)
        center = (boundary + maxrange) / 2
        diff = maxrange - boundary
        dist = torch.norm(diff, dim=2, p=2).unsqueeze(-1)

        semi_major = dist / 2 + self.cfg.safe_dist
        semi_minor = self.cfg.ellipse_width * torch.ones_like(semi_major)
        a = torch.diag_embed(torch.stack([1 / semi_major.pow(2), 1 / semi_minor.pow(2)], dim=-1).squeeze(-2))

        angle = torch.atan2(diff[..., 1], diff[..., 0])
        r = rotz_2d(angle)
        a_rot = torch.einsum('bkmn,bknv,bksv->bkms', r, a, r)

        def ellipse(x, robot_indices):
            if robot_indices is not None:
                selected_center = center[robot_indices, :, :]
                selected_a_rot = a_rot[robot_indices, :, :, :]
                x_expanded = x.unsqueeze(2).expand(-1, -1, selected_center.shape[1], -1)
                q = x_expanded - torch.repeat_interleave(selected_center.unsqueeze(0), x.shape[0], dim=0)
                return torch.einsum('bijk,ijkl,bijl->bij', q, selected_a_rot, q) - 1
            else:
                x_expanded = x.unsqueeze(2).expand(-1, -1, center.shape[1], -1)
                q = x_expanded - torch.repeat_interleave(center.unsqueeze(0), x.shape[0], dim=0)
                return torch.einsum('bijk,ijkl,bijl->bij', q, a_rot, q) - 1

        return ellipse

    def _make_circle(self, center):
        max_range_sq = self.lidar.lidar_cfg.max_range ** 2
        def circle(x, robot_indices):
            if robot_indices is not None:
                selected_center = center[:, robot_indices, :]
                return (max_range_sq - torch.norm((x - torch.repeat_interleave(selected_center,
                                                                                 x.shape[0], dim=0)), dim=2) ** 2)
            else:
                return max_range_sq - torch.norm((x - torch.repeat_interleave(center,x.shape[0], dim=0)),dim=2) ** 2
        return circle


class SVMCBF(CBFSynthesis):
    def __init__(self, lidar, cfg):
        super().__init__(lidar, cfg)

    def update(self, lidar):
        pass
