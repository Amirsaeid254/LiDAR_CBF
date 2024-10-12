from math import comb
import torch
from LiDAR_CBF.utils.utils import mod


def create_smooth_function(cfg, period):
    def smooth_step(r):
        return lambda t: torch.where(
            t <= 0,
            torch.zeros_like(t, requires_grad=t.requires_grad),
            torch.where(
                t >= 1,
                torch.ones_like(t, requires_grad=t.requires_grad),
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

    def sinusoidal_step(r):
        assert r <= 2, 'Please choose SmoothStep function for relative degree higher than 2'
        return lambda t: torch.where(
            t < 0,
            torch.zeros_like(t),
            torch.where(
                t > 1,
                torch.ones_like(t),
                (t) - ((2 * torch.pi) ** -1) * torch.sin(2 * torch.pi * t),
            ),
        )

    smooth_functions = {
        'SmoothStep': smooth_step,
        'SinusoidalStep': sinusoidal_step
    }

    assert cfg.smooth_function in smooth_functions, "smooth function method not implemented"

    smooth_func = smooth_functions[cfg.smooth_function](cfg.pos_barrier_rel_deg)

    def periodic_function(t):
        return smooth_func((torch.remainder(t, period) * cfg.nu) / period)
        # return smooth_func((mod(t, period) * cfg.nu) / period)

    return periodic_function
