import torch
from hocbf_composition.utils.dynamics import AffineInControlDynamics

class UnicycleDynamics(AffineInControlDynamics):
    def _f(self, x):
        return torch.stack([x[:, 2] * torch.cos(x[:, 3]),
                            x[:, 2] * torch.sin(x[:, 3]),
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.vstack([torch.zeros(2, 2, dtype=torch.float64),
                              torch.eye(2, dtype=torch.float64)])
                ).repeat(x.shape[0], 1, 1)

    def get_pos(self, x):
        return x[..., 0:2]

    def get_rot(self, x):
        return x[..., -1]





