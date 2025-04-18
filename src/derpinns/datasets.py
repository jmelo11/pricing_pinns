from torch import nn
import torch
from torch.utils.data import Dataset
from derpinns.collocations import *


class SampledDataset(Dataset):
    """
        Generates a dataset of collocations points.
    """

    def __init__(self, params: OptionParameters, interior_samples: int, initial_samples: int, boundary_samples: int, sampler: str, dtype: torch.dtype, device: torch.device, verbose: bool = False, seed=None):
        self.params = params
        x, y, mask = generate_dataset(
            params, interior_samples, initial_samples, boundary_samples, sampler, seed)

        self.x = torch.tensor(
            x, dtype=dtype, device=device, requires_grad=True)

        self.y = torch.tensor(y, dtype=dtype, device=device)
        self.mask = torch.tensor(mask, dtype=torch.bool, device=device)

        if verbose:
            import matplotlib.pyplot as plt
            print(self.mask[:, 1])
            interior_mask = self.mask[:, 0] == 1
            interior_points = self.x[interior_mask].cpu().detach().numpy()
            print(interior_mask, interior_points)
            plt.scatter(interior_points[:, 0], interior_points[:, 1])
            plt.show()

            print("Shapes:")
            print(f"x: {self.x.shape}")
            print(f"y: {self.y.shape}")
            print(f"mask: {self.mask.shape}")
            for i in range(self.mask.shape[1]):
                print(f"mask[:, {i}].sum(): {self.mask[:, i].sum().item()}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx, :]


class SampledDatasetWithPINNBoundary(Dataset):
    """
        Takes advantage of the fact that for the lower boundary, the solution is the same as the n-1th asset case.
    """

    def __init__(self, pinn: nn.Module, params: OptionParameters, interior_samples: int, initial_samples: int, boundary_samples: int, sampler: str, dtype: torch.dtype, device: torch.device, verbose: bool = False):
        self.params = params
        x, y, mask = generate_dataset(
            params, interior_samples, initial_samples, boundary_samples, sampler)

        # Compute the boundary values using the PINN model
        # for the lower boundary (n-1th asset case)
        for i in range(params.n_assets):
            lb_idx = 2*i + 2
            lb_mask = self.mask[:, lb_idx]  # lower boundary for the i-th asset
            lb_x = self.x[lb_mask]
            # remove ith asset from the input
            lb_x = np.cat([lb_x[:, :i], lb_x[:, i+1:]], dim=1)
            lb_x = torch.tensor(lb_x, dtype=dtype, device=device)
            lb_y = pinn(lb_x).cpu().detach().numpy()
            # set the lower boundary value for the i-th asset
            y[lb_mask] = lb_y

        self.x = torch.tensor(
            x, dtype=dtype, device=device, requires_grad=True)

        self.y = torch.tensor(y, dtype=dtype, device=device)
        self.mask = torch.tensor(mask, dtype=torch.bool, device=device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx, :]
