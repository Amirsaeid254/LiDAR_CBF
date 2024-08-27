from math import comb
import torch
from LiDAR_CBF.utils.utils import mod


class SmoothFunction():
    def __init__(self, cfg, period):
        self.period = period
        relative_degree = cfg.pos_barrier_rel_deg + 1
        assert hasattr(self, cfg.smooth_function), "smooth function method not implemented"
        self.smooth_function = getattr(self, cfg.smooth_function)(relative_degree)
        self.periodic_function = self.make_periodic(cfg.nu)

    def __call__(self, t):
        return self.periodic_function(t)

    def make_periodic(self, nu):
        return lambda t: self.smooth_function((mod(t, self.period) * nu) / self.period)

    def SmoothStep(self, r):
        return lambda t: torch.where(
            t <= 0,
            torch.zeros_like(t),
            torch.where(
                t >= 1,
                torch.ones_like(t),
                (t) ** (r + 1) * torch.sum(
                    torch.stack(
                        [
                            comb(r + j, j) * comb(2 * r + 1, r - j) * torch.pow(-t, j)
                            for j in range(r + 1)
                        ],
                        dim=0,
                    ),
                    dim=0,
                ),
            ),
        )

    def SinusoidalStep(self, r):
        assert r <= 2, 'Please choose SmoothStep function for relatrive degree higher than 2'
        self.smooth_function = lambda t: torch.where(
            t < 0,
            torch.zeros_like(t),
            torch.where(
                t > (1),
                torch.ones_like(t),
                (t) - ((2 * torch.pi) ** -1) * torch.sin(2 * torch.pi * t),
            ),
        )
