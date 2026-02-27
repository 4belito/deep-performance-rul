import torch


def inv_softplus(x, eps=1e-6):
    x = torch.as_tensor(x)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(torch.clamp(x, min=eps))))
