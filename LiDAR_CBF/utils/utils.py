
import torch



def rotz_2d(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((c, -s, s, c), dim=2).view(theta.shape[0], theta.shape[1], 2, 2)


def mod(a, m):
    # This function replicates the behavior of MATLAB's mod function in PyTorch.
    remainder = a - m * torch.floor(a / m + 1e-9)
    return remainder


def vectorize_input(x):
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(0)

    return x


def floor_div(input, K):
    rem = torch.remainder(input, K)
    out = (input - rem) / K
    return out